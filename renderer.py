import trimesh
import pyrender
import numpy as np


class Renderer(object):
    def __init__(self, fx, fy, img_H, img_W):
        self.reset(fx, fy, img_H, img_W)
        
    def reset(self, fx, fy, img_H, img_W):
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_W, viewport_height=img_H, point_size=1.0
        )
        self.fx = fx
        self.fy = fy
        self.cx = img_W // 2
        self.cy = img_H // 2
        
        self.scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3)
        )
        # Add lights
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        self.scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        self.scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        self.scene.add(light, pose=light_pose)
        
        # Mesh material
        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            # baseColorFactor=(1.0, 0.5, 0.5, 0.8)
        )
        self.mesh_color_list = [
            [1.0, 0.5, 0.5, 0.7],
            [0.5, 0.5, 1.0, 0.7],
            [0.5, 1.0, 0.5, 0.7]
        ] # maximum 3
    
    def __call__(self, vertices_list, faces_list, img,
                 cam_R = np.eye(3), cam_t=np.zeros(3), wireframe=False):
        # Add each mesh
        mesh_nodes = []
        for vertices, faces, mesh_color in zip(vertices_list, faces_list, self.mesh_color_list):
            mesh = trimesh.Trimesh(vertices, faces)
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0]
            )
            mesh.apply_transform(rot)
            self.material.baseColorFactor = mesh_color
            mesh = pyrender.Mesh.from_trimesh(mesh, material=self.material)
            mesh_nodes.append(self.scene.add(mesh))
                
        # Add camera
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = cam_R
        cam_pose[:3, 3] = cam_t
        camera = pyrender.IntrinsicsCamera(
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )
        cam_node = self.scene.add(camera, pose=cam_pose)

        # Projection
        if wireframe:
            render_flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.ALL_WIREFRAME
        else:
            render_flags = pyrender.RenderFlags.RGBA
        color, _ = self.renderer.render(self.scene, flags=render_flags)
                
        # Overlay on image
        alpha_channel = color[:, :, 3] / 255.0
        mesh_color = color[:, :, :3]
        alpha_mask = np.dstack([alpha_channel, alpha_channel, alpha_channel])
        H, W, _ = img.shape
        background_subsection = img[:H, :W]
        composite = background_subsection * (1-alpha_mask) + mesh_color * alpha_mask
        img[:H, :W] = composite
        
        for i in range(len(mesh_nodes)):
            self.scene.remove_node(mesh_nodes[i])
        self.scene.remove_node(cam_node)
        
        return img
