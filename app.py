INPUT_MOVIE_FILE = None
OUTPUT_CSV_FILE = None
bVisualize = False

PREVIEW_WINDOW_SIZE = 480
MAX_FRAMES = -1

import cv2
import keyboard
import numpy as np
import open3d as o3d
import pygame
from transforms3d.axangles import axangle2mat
import sys

import config
#from capture import OpenCVCapture
from hand_mesh import HandMesh
from kinematics import mpii_to_mano
from utils import OneEuroFilter, imresize
from wrappers import ModelPipeline
from utils import *

#def live_application(capture):
def live_application():
  """
  Launch an application that reads from a webcam and estimates hand pose at
  real-time.

  The captured hand must be the right hand, but will be flipped internally
  and rendered.

  Parameters
  ----------
  capture : object
    An object from `capture.py` to read capture stream from.
  """
  ############ output visualization ############
  view_mat = axangle2mat([1, 0, 0], np.pi) # align different coordinate systems
  window_size = PREVIEW_WINDOW_SIZE #1080

  hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)

  if bVisualize:
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    mesh.vertices = \
      o3d.utility.Vector3dVector(np.matmul(view_mat, hand_mesh.verts.T).T * 1000)
    mesh.compute_vertex_normals()

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(
      width=window_size + 1, height=window_size + 1,
      window_name='Minimal Hand - output'
    )
    viewer.add_geometry(mesh)

    view_control = viewer.get_view_control()
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = cam_params.extrinsic.copy()
    extrinsic[0:3, 3] = 0
    cam_params.extrinsic = extrinsic
    cam_params.intrinsic.set_intrinsics(
      window_size + 1, window_size + 1, config.CAM_FX, config.CAM_FY,
      window_size // 2, window_size // 2
    )
    view_control.convert_from_pinhole_camera_parameters(cam_params)
    view_control.set_constant_z_far(1000)

    render_option = viewer.get_render_option()
    render_option.load_from_json('./render_option.json')
    viewer.update_renderer()

    ############ input visualization ############
    pygame.init()
    display = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption('Minimal Hand - input')

  ############ misc ############
  mesh_smoother = OneEuroFilter(4.0, 0.0)
  clock = pygame.time.Clock()
  model = ModelPipeline()

  # Video read
  cap = cv2.VideoCapture(INPUT_MOVIE_FILE)
  totalFrameNum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  fout = open(OUTPUT_CSV_FILE,mode='w')
  frameNo=0

  print('Start processing.')
  try:
   while cap.isOpened():  # True
    ret_val, inimg = cap.read()
    if not ret_val: continue
    frameNo=frameNo+1
    if frameNo==MAX_FRAMES : break

    # Fit input (assuming square) image into 640x480 frame_large image
    img = cv2.cvtColor(inimg,cv2.COLOR_BGR2RGB)
    #img = cv2.flip(img,1)
    img = cv2.resize(img,(480,480))
    frame_large = cv2.copyMakeBorder(img,0,0,80,80,cv2.BORDER_CONSTANT,value=[0,0,0])

    if frame_large.shape[0] > frame_large.shape[1]:
      margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
      frame_large = frame_large[margin:-margin]
    else:
      margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
      frame_large = frame_large[:, margin:-margin]


    frame_large = np.flip(frame_large, axis=1).copy()
    frame = imresize(frame_large, (128, 128))

    _, theta_mpii = model.process(frame)
    theta_mano = mpii_to_mano(theta_mpii)

    v,joint_xyz = hand_mesh.set_abs_quat(theta_mano)

    #print('Joint pos:',joint_xyz)
    line = ','.join( list(map(lambda x: '%f,%f,%f' %(x[0],x[1],x[2]),joint_xyz)) )
    fout.write(line+"\n")
    if frameNo%50 == 0 :
      print("\r %d/%d frames processed." % (frameNo,totalFrameNum) ,end='')

    if not bVisualize: continue

    #v *= 2 # for better visualization
    v = v * 1000 + np.array([0, 0, 400])
    #v = mesh_smoother.process(v)
    mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, v.T).T)
    mesh.paint_uniform_color(config.HAND_COLOR)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    # for some version of open3d you may need `viewer.update_geometry(mesh)`
    viewer.update_geometry()

    viewer.poll_events()

    display.blit(
      pygame.surfarray.make_surface(
        np.transpose(
          imresize(frame_large, (window_size, window_size)
        ), (1, 0, 2))
      ),
      (0, 0)
    )
    pygame.display.update()

    #if keyboard.is_pressed("esc"):
    #  break

    clock.tick(30)
  except KeyboardInterrupt:
    print('Ctrl-c catched')

  fout.close()
  print("Completed.")

if __name__ == '__main__':

  usage = 'Usage: python {} INPUT_MOVIE_FILE OUTPUT_CSV_FILE [-viz]'.format(__file__)

  if len(sys.argv) < 3:
    print( usage )
    sys.exit(1)
    #  return usage

  INPUT_MOVIE_FILE = sys.argv[1]
  OUTPUT_CSV_FILE = sys.argv[2]

  bVisualize = (len(sys.argv)==4)

  live_application()
  #live_application(OpenCVCapture())
