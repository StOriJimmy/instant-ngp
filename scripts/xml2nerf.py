#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil

from xml.etree import ElementTree


def parse_args():
	parser = argparse.ArgumentParser(description="convert a context capture xml export to nerf format transforms.json")

	parser.add_argument("--xml", default="xml", help="input path of the xml")
	parser.add_argument("--aabb_scale", default=16, choices=["1", "2", "4", "8", "16"], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
	parser.add_argument("--skip_early", default=0, help="skip this many images from the start")
	parser.add_argument("--keep_coords", action="store_true",
						help="keep transforms.json in original frame of reference (this will avoid reorienting and repositioning the scene for preview and rendering)")
	parser.add_argument("--out", default="transforms_", help="output path prefix, eg: transforms_")
	args = parser.parse_args()
	return args


def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm


def qvec2rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2
		]
	])


def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom


if __name__ == "__main__":
	args = parse_args()

	AABB_SCALE = int(args.aabb_scale)
	SKIP_EARLY = int(args.skip_early)
	XML_PATH = args.xml
	OUT_PATH_PREFIX = args.out
	print(f"outputting to {OUT_PATH_PREFIX}...")

	tree = ElementTree.parse(XML_PATH)
	root = tree.getroot()

	Block = tree.find('Block')
	print(Block)

	Photogroups = Block.find('Photogroups')
	print(Photogroups)

	Photogroups_list = Photogroups.findall('Photogroup')
	photogroup_id = 0

	bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
	flip_mat = np.array([
		[1, 0, 0, 0],
		[0, -1, 0, 0],
		[0, 0, -1, 0],
		[0, 0, 0, 1]
	])
	up = np.zeros(3)

	out_list = {
		"out": [],
		"path": []
	}

	for Photogroup in Photogroups_list:
		OUTPUT_PATH = OUT_PATH_PREFIX + str(photogroup_id) + ".json"

		print(f"outputting to {OUTPUT_PATH}")

		ImageDimensions = Photogroup.find('ImageDimensions')

		width = int(ImageDimensions.find('Width').text)
		height = int(ImageDimensions.find('Height').text)

		fl_x = 0.0
		fl_y = 0.0
		FocalLengthPixels = Photogroup.find('FocalLengthPixels')
		if FocalLengthPixels is None:
			FocalLength = Photogroup.find('FocalLength')
			SensorSize = Photogroup.find('SensorSize')
			if FocalLength is None or SensorSize is None:
				continue
			else:
				fl_x = float(FocalLength.text) * max(width, height) / float(SensorSize.text)
				fl_y = fl_x
		else:
			fl_x = float(FocalLengthPixels.text)
			fl_y = float(FocalLengthPixels.text)

		PrincipalPoint = Photogroup.find('PrincipalPoint')
		if PrincipalPoint is None:
			continue
		cx = float(PrincipalPoint.find('x').text)
		cy = float(PrincipalPoint.find('y').text)

		Distortion = Photogroup.find('Distortion')
		if Distortion is None:
			continue
		k1 = float(Distortion.find('K1').text)
		k2 = float(Distortion.find('K2').text)
		k3 = float(Distortion.find('K3').text)
		p0 = float(Distortion.find('P1').text)
		p1 = float(Distortion.find('P2').text)

		angle_x = math.atan(width / (fl_x * 2)) * 2
		angle_y = math.atan(height / (fl_y * 2)) * 2
		fov_x = angle_x * 180 / math.pi
		fov_y = angle_y * 180 / math.pi

		print(
			f"camera:\n\tres={width, height}\n\tcenter={cx, cy}\n\tfocal={fl_x, fl_y}\n\tfov={fov_x, fov_y}\n\tk={k1, k2} p={p0, p1} ")

		out = {
			"camera_angle_x": angle_x,
			"camera_angle_y": angle_y,
			"fl_x": fl_x,
			"fl_y": fl_y,
			"k1": k1,
			"k2": k2,
			"p1": p0,
			"p2": p1,
			"cx": cx,
			"cy": cy,
			"w": width,
			"h": height,
			"aabb_scale": AABB_SCALE,
			"frames": [],
		}

		for Photo in Photogroup.findall('Photo'):
			ImagePath = Photo.find('ImagePath').text
			Id = int(Photo.find('Id').text)
			Component = int(Photo.find('Component').text)
			Pose = Photo.find('Pose')
			Rotation = Pose.find('Rotation')

			M_00 = float(Rotation.find('M_00').text)
			M_01 = float(Rotation.find('M_01').text)
			M_02 = float(Rotation.find('M_02').text)
			M_10 = float(Rotation.find('M_10').text)
			M_11 = float(Rotation.find('M_11').text)
			M_12 = float(Rotation.find('M_12').text)
			M_20 = float(Rotation.find('M_20').text)
			M_21 = float(Rotation.find('M_21').text)
			M_22 = float(Rotation.find('M_22').text)

			Center = Pose.find('Center')
			CX = float(Center.find('x').text)
			CY = float(Center.find('y').text)
			CZ = float(Center.find('z').text)

			R = np.array([
				[M_00, M_01, M_02],
				[M_10, M_11, M_12],
				[M_20, M_21, M_22]
			])
			print("R:", R)
			C = np.array([CX, CY, CZ])
			print("C:", C)

			T = -np.dot(R, C)

			print("T:", T)

			vv = np.concatenate([R, T.reshape([3,1])], 1)
			print(vv)

			m = np.concatenate([vv, bottom], 0)
			print("m:", m)

			sp = sharpness(ImagePath)
			print(ImagePath, "sharpness=", sp)

			c2w = np.linalg.inv(m)

			if not args.keep_coords:
				c2w[0:3, 2] *= -1  # flip the y and z axis
				c2w[0:3, 1] *= -1
				c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
				c2w[2, :] *= -1  # flip whole world upside down

				up += c2w[0:3, 1]

			if args.keep_coords:
				c2w = np.matmul(c2w, flip_mat)

			frame={"file_path":ImagePath,"sharpness":sp,"transform_matrix":c2w}
			out["frames"].append(frame)

		nframes = len(out["frames"])
		print(nframes, "frames")

		photogroup_id = photogroup_id + 1
		out_list["out"].append(out)
		out_list["path"].append(OUTPUT_PATH)

	for i in range(photogroup_id):
		OUTPUT_PATH = out_list["path"][i]
		print(f"writing {OUTPUT_PATH}")

		out = out_list["out"][i]
		if args.keep_coords:
			for f in out["frames"]:
				f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat)
		else:
			# don't keep coords - reorient the scene to be easier to work with

			up = up / np.linalg.norm(up)
			print("up vector was", up)
			R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
			R = np.pad(R, [0, 1])
			R[-1, -1] = 1

			for f in out["frames"]:
				f["transform_matrix"] = np.matmul(R, f["transform_matrix"])  # rotate up to be the z axis

			# find a central point they are all looking at
			print("computing center of attention...")
			totw = 0.0
			totp = np.array([0.0, 0.0, 0.0])
			for f in out["frames"]:
				mf = f["transform_matrix"][0:3, :]
				for g in out["frames"]:
					mg = g["transform_matrix"][0:3, :]
					p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
					if w > 0.01:
						totp += p * w
						totw += w
			totp /= totw
			print(totp)  # the cameras are looking at totp
			for f in out["frames"]:
				f["transform_matrix"][0:3, 3] -= totp

			avglen = 0.
			for f in out["frames"]:
				avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
			avglen /= nframes
			print("avg camera distance from origin", avglen)
			for f in out["frames"]:
				f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"

		for f in out["frames"]:
			f["transform_matrix"] = f["transform_matrix"].tolist()

		with open(OUTPUT_PATH, "w") as outfile:
			json.dump(out, outfile, indent=2)

