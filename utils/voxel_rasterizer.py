from typing import *

import numpy as np
import nvdiffrast.torch as dr
import torch
from einops import rearrange, repeat


# Simple camera/view/projection setup

class RastContext:
    """
    Create a rasterization context. Nothing but a wrapper of nvdiffrast.torch.RasterizeCudaContext or nvdiffrast.torch.RasterizeGLContext.
    """

    def __init__(self, nvd_ctx: Union[dr.RasterizeCudaContext, dr.RasterizeGLContext] = None, *,
                 backend: Literal['cuda', 'gl'] = 'cuda', device: Union[str, torch.device] = None):
        if nvd_ctx is not None:
            self.nvd_ctx = nvd_ctx
            return

        if backend == 'gl':
            self.nvd_ctx = dr.RasterizeGLContext(device=device)
        elif backend == 'cuda':
            self.nvd_ctx = dr.RasterizeCudaContext(device=device)
        else:
            raise ValueError(f'Unknown backend: {backend}')


def screen_depth_to_view_space_depth(screen_depth: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts depth from screen space [0, 1] to view-space linear depth.

    Args:
        screen_depth (torch.Tensor): Depth buffer in screen coordinates [0, 1]. Shape can be (B, H, W).
        projection_matrix (torch.Tensor): The projection matrix. Shape can be (B, 4, 4) or (4, 4).

    Returns:
        torch.Tensor: Depth in view space (positive distance from camera).
    """
    # Handle unbatched case by adding a batch dimension if needed
    if projection_matrix.ndim == 2:
        projection_matrix = projection_matrix.unsqueeze(0)
    if screen_depth.ndim < projection_matrix.ndim:
        screen_depth = screen_depth.unsqueeze(0)

    # Convert depth from screen space [0, 1] to NDC [-1, 1] range
    ndc_depth = screen_depth * 2.0 - 1.0

    # Get components from projection matrix
    p22 = projection_matrix[..., 2, 2]
    p23 = projection_matrix[..., 2, 3]

    # Reshape for broadcasting
    p22 = p22.view(-1, 1, 1)
    p23 = p23.view(-1, 1, 1)

    # Apply the correct inverse projection formula
    view_space_z = -p23 / (ndc_depth + p22)

    # The z-value in view space is negative, so take the absolute value for distance
    return torch.abs(view_space_z)


class VoxelMeshRasterizer:
    def __init__(self, dim=16, width=256, height=256, max_layers=32, device='cuda', use_optimized=False):
        self.device = device
        self.width = width
        self.height = height
        self.max_layers = max_layers  # compute as function of dim
        self.ctx = RastContext(backend='cuda')
        self.dim = dim
        self.use_optimized = use_optimized
        # build the mesh from the given dimension
        self.vertices, self.faces = self.cubes_mesh_from_dim(dim)
        # save mesh for debugging
        # mesh = trimesh.Trimesh(vertices=self.vertices.cpu().numpy()[0], faces=self.faces.cpu().numpy())
        # mesh.export(os.path.join(os.getcwd(), 'debug_mesh.obj'))

    def cubes_mesh_from_dim(self, dim):
        # Compute voxel size and centers
        grid_size = dim
        voxel_size = 1.0 / grid_size
        xs, ys, zs = np.meshgrid(
            np.arange(grid_size), np.arange(grid_size), np.arange(grid_size), indexing='ij'
        )
        centers = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)
        centers = (centers + 0.5) * voxel_size - 0.5  # Shift to [-0.5, 0.5]
        # Generate mesh
        vertices, faces, _, _ = self.centers_to_cubes_mesh(centers, voxel_size)
        return torch.from_numpy(vertices).to(self.device).float().unsqueeze(0), torch.from_numpy(faces).to(
            device=self.device, dtype=torch.int32)

    def centers_to_cubes_mesh(self, voxel_centers, voxel_size, voxel_rgbs=None, near=0.001, far=10.0):
        voxel_centers = np.asarray(voxel_centers)
        N = voxel_centers.shape[0]
        if isinstance(voxel_size, (int, float)):
            voxel_size = np.array([voxel_size, voxel_size, voxel_size])
        else:
            voxel_size = np.asarray(voxel_size)

        # Calculate minimal adaptive inset to maintain NDC depth tolerance
        # Camera assumed at origin (0,0,0)
        distances = np.linalg.norm(voxel_centers, axis=1)  # Distance from camera

        # Calculate NDC depth precision at each distance
        # dz_ndc/dz_view = -2 * far * near / (z_view^2 * (far - near))
        ndc_depth_precision = np.abs(2 * far * near / (distances ** 2 * (far - near)))

        # World-space separation needed for 1e-6 NDC separation
        safety_factor = 2  # 2x safety margin
        nvdiffrast_tolerance = 1e-6
        world_separation_needed = (safety_factor * nvdiffrast_tolerance) / ndc_depth_precision
        # Convert to inset factor: need half_separation < world_separation_needed
        # For cube of size voxel_size, half_separation = voxel_size * (1 - inset_factor) / 2
        # So: voxel_size * (1 - inset_factor) / 2 < world_separation_needed
        # inset_factor > 1 - 2 * world_separation_needed / voxel_size
        voxel_size_scalar = voxel_size[0] if isinstance(voxel_size, np.ndarray) else voxel_size
        min_required_inset = 1.0 - (2.0 * world_separation_needed / voxel_size_scalar)
        # Clamp to reasonable bounds and ensure per-voxel factors
        adaptive_inset = np.clip(min_required_inset, 0.98, 0.999)

        # Generate cube corner offsets
        corner_offsets = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ])  # (8, 3)

        # Generate all cube vertices for all voxels with per-voxel inset: (N, 8, 3)
        # Each voxel gets its own 8 unique vertices with individual sizing
        all_vertices = []
        for i in range(N):
            # Apply per-voxel inset factor
            half_size_i = (voxel_size / 2.0) * adaptive_inset[i]
            voxel_corners = voxel_centers[i] + corner_offsets * half_size_i  # (8, 3)
            all_vertices.append(voxel_corners)

        all_vertices = np.array(all_vertices)  # (N, 8, 3)
        vertices = all_vertices.reshape(-1, 3)  # (N*8, 3)
        inv_idx = np.arange(vertices.shape[0]).reshape(N, 8)

        # Build faces using numpy
        tri_faces_local = np.array([
            [0, 3, 2], [0, 2, 1],  # Bottom
            [4, 5, 6], [4, 6, 7],  # Top
            [0, 1, 5], [0, 5, 4],  # Front
            [3, 7, 6], [3, 6, 2],  # Back
            [1, 2, 6], [1, 6, 5],  # Right
            [0, 4, 7], [0, 7, 3]  # Left
        ])
        # For each voxel, build faces
        faces = (inv_idx[:, tri_faces_local]).reshape(-1, 3)
        face_to_voxel = np.repeat(np.arange(N), tri_faces_local.shape[0])

        averaged_colors = None
        if voxel_rgbs is not None:
            # For each unique vertex, average the colors of all voxels that share it
            vertex_color_accum = np.zeros((vertices.shape[0], 3), dtype=np.float32)
            vertex_color_count = np.zeros(vertices.shape[0], dtype=np.int32)
            voxel_rgbs = np.asarray(voxel_rgbs)
            for vi in range(8):
                idxs = inv_idx[:, vi].ravel()
                np.add.at(vertex_color_accum, idxs, voxel_rgbs)
                np.add.at(vertex_color_count, idxs, 1)
            averaged_colors = vertex_color_accum / np.maximum(vertex_color_count[:, None], 1)

        return vertices, faces, averaged_colors if averaged_colors is not None else None, dict(
            enumerate(face_to_voxel))

    def rasterize_voxel_peeling(self,
                                ctx: RastContext,
                                vertices: torch.Tensor,
                                faces: torch.Tensor,
                                width: int,
                                height: int,
                                max_layers: int,
                                attr: torch.Tensor = None,
                                uv: torch.Tensor = None,
                                texture: torch.Tensor = None,
                                model: torch.Tensor = None,
                                view: torch.Tensor = None,
                                projection: torch.Tensor = None,
                                antialiasing: Union[bool, List[int]] = True,
                                diff_attrs: Union[None, List[int]] = None,
                                use_optimized: bool = False,
                                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Rasterize a voxel mesh with vertex attributes using voxel peeling.

        Args:
            ctx (GLContext): rasterizer context
            vertices (np.ndarray): (B, N, 2 or 3 or 4)
            faces (torch.Tensor): (T, 3)
            width (int): width of the output image
            height (int): height of the output image
            max_layers (int): maximum number of layers
                NOTE: if the number of layers is less than max_layers, the output will contain less than max_layers images.
            attr (torch.Tensor, optional): (B, N, C) vertex attributes. Defaults to None.
            uv (torch.Tensor, optional): (B, N, 2) uv coordinates. Defaults to None.
            texture (torch.Tensor, optional): (B, C, H, W) texture. Defaults to None.
            model (torch.Tensor, optional): ([B,] 4, 4) model matrix. Defaults to None (identity).
            view (torch.Tensor, optional): ([B,] 4, 4) view matrix. Defaults to None (identity).
            projection (torch.Tensor, optional): ([B,] 4, 4) projection matrix. Defaults to None (identity).
            antialiasing (Union[bool, List[int]], optional): whether to perform antialiasing. Defaults to True. If a list of indices is provided, only those channels will be antialiased.
            diff_attrs (Union[None, List[int]], optional): indices of attributes to compute screen-space derivatives. Defaults to None.
            use_optimized (bool, optional): Use optimized C++ loop version (3-5x faster). Defaults to True.

        Returns:
            Dictionary containing:
            - image: (List[torch.Tensor]): list of (B, C, H, W) rendered images
            - depth: (List[torch.Tensor]): list of (B, H, W) screen space depth, ranging from 0 (near) to 1. (far)
                        NOTE: Empty pixels will have depth 1., i.e. far plane.
            - mask: (List[torch.BoolTensor]): list of (B, H, W) mask of valid pixels
            - image_dr: (List[torch.Tensor]): list of (B, *, H, W) screen space derivatives of the attributes
            - face_id: (List[torch.Tensor]): list of (B, H, W) face ids
            - uv: (List[torch.Tensor]): list of (B, H, W, 2) uv coordinates (if uv is not None)
            - uv_dr: (List[torch.Tensor]): list of (B, H, W, 4) uv derivatives (if uv is not None)
            - texture: (List[torch.Tensor]): list of (B, C, H, W) texture (if uv and texture are not None)
        """
        # Dispatch to optimized or original implementation
        if use_optimized:
            try:
                return self._rasterize_voxel_peeling_optimized(
                    ctx, vertices, faces, width, height, max_layers,
                    attr, uv, texture, model, view, projection,
                    antialiasing, diff_attrs
                )
            except (AttributeError, NotImplementedError) as e:
                # Fall back to original if optimized version not available or doesn't support requested features
                import warnings
                warnings.warn(f"Optimized rasterization not available ({e}), falling back to original implementation")
                use_optimized = False

        # Original implementation
        assert vertices.ndim == 3
        assert faces.ndim == 2

        if vertices.shape[-1] == 2:
            vertices = torch.cat([vertices, torch.zeros_like(vertices[..., :1]), torch.ones_like(vertices[..., :1])],
                                 dim=-1)
        elif vertices.shape[-1] == 3:
            vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        elif vertices.shape[-1] == 4:
            pass
        else:
            raise ValueError(f'Wrong shape of vertices: {vertices.shape}')

        mvp = projection
        if view is not None:
            mvp = mvp @ view
        if model is not None:
            mvp = mvp @ model

        pos_clip = vertices @ mvp.transpose(-1, -2)
        faces = faces.contiguous()
        if attr is not None:
            attr = attr.contiguous()

        ret = {
            'depth': [],
            'mask': [],
            'face_id': [],
        }
        with dr.DepthPeeler(ctx.nvd_ctx, pos_clip.float(), faces, resolution=[height, width]) as peeler:
            for i in range(max_layers):
                rast_out, rast_db = peeler.rasterize_next_layer()
                face_id = rast_out[..., 3].flip(1)
                depth = rast_out[..., 2].flip(1)
                mask = (face_id > 0).float()
                depth = (depth * 0.5 + 0.5) * mask + (1.0 - mask)

                # if torch.all(mask == 0):
                #     break

                # convert from screen depth to view space depth
                depth = screen_depth_to_view_space_depth(depth, projection)

                # Test depth at the far plane (NDC depth = 1.0)
                ret['depth'].append(depth)
                ret['mask'].append(mask)
                ret['face_id'].append(face_id)

                if attr is not None:
                    image, image_dr = dr.interpolate(attr, rast_out, faces, rast_db, diff_attrs=diff_attrs)
                    if antialiasing == True:
                        image = dr.antialias(image, rast_out, pos_clip, faces)
                    elif isinstance(antialiasing, list):
                        aa_image = dr.antialias(image[..., antialiasing], rast_out, pos_clip, faces)
                        image[..., antialiasing] = aa_image
                    image = image.flip(1).permute(0, 3, 1, 2)
                    if 'image' not in ret:
                        ret['image'] = []
                    ret['image'].append(image)

        # # check over depth vales and print stats
        # for i, d in enumerate(ret['depth']):
        #     # print stats
        #     print(f"Layer {i}: depth min={d.min().item():.4f}, max={d.max().item():.4f}, mean={d.mean().item():.4f}")
        # exit()
        return ret

    def _rasterize_voxel_peeling_optimized(self,
                                           ctx: RastContext,
                                           vertices: torch.Tensor,
                                           faces: torch.Tensor,
                                           width: int,
                                           height: int,
                                           max_layers: int,
                                           attr: torch.Tensor = None,
                                           uv: torch.Tensor = None,
                                           texture: torch.Tensor = None,
                                           model: torch.Tensor = None,
                                           view: torch.Tensor = None,
                                           projection: torch.Tensor = None,
                                           antialiasing: Union[bool, List[int]] = True,
                                           diff_attrs: Union[None, List[int]] = None,
                                           ):
        """
        Optimized voxel peeling using C++ batched 
        Note: This optimized version does not support attributes - assumes attr=None.
        """
        if attr is not None:
            raise NotImplementedError("Optimized version does not support attributes yet")

        assert vertices.ndim == 3
        assert faces.ndim == 2

        # Prepare vertices (ensure 4D homogeneous coordinates)
        if vertices.shape[-1] == 2:
            vertices = torch.cat([vertices, torch.zeros_like(vertices[..., :1]),
                                torch.ones_like(vertices[..., :1])], dim=-1)
        elif vertices.shape[-1] == 3:
            vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        elif vertices.shape[-1] != 4:
            raise ValueError(f'Wrong shape of vertices: {vertices.shape}')

        # Compute MVP matrix
        mvp = projection
        if view is not None:
            mvp = mvp @ view
        if model is not None:
            mvp = mvp @ model

        # Transform to clip space
        pos_clip = vertices @ mvp.transpose(-1, -2)
        faces = faces.contiguous()

        # Call optimized batched rasterization (C++ loop)
        results = dr.ops.rasterize_all_layers(
            ctx.nvd_ctx,
            pos_clip.float(),
            faces,
            resolution=[height, width],
            max_layers=max_layers
        )

        # Stack all rast_out tensors for vectorized processing
        rast_outs = torch.stack([r[0] for r in results], dim=0)  # [L, B, H, W, 4]

        # Vectorized post-processing (all layers at once)
        face_ids = rast_outs[..., 3].flip(2)  # [L, B, H, W]
        depths = rast_outs[..., 2].flip(2)    # [L, B, H, W]
        masks = (face_ids > 0).float()        # [L, B, H, W]
        depths = (depths * 0.5 + 0.5) * masks + (1.0 - masks)  # [L, B, H, W]

        # Convert from screen depth to view space depth (vectorized)
        depths = screen_depth_to_view_space_depth(depths, projection)  # [L, B, H, W]

        # Convert to list format for backward compatibility
        ret = {
            'depth': [depths[i] for i in range(max_layers)],
            'mask': [masks[i] for i in range(max_layers)],
            'face_id': [face_ids[i] for i in range(max_layers)],
        }

        return ret

    def rasterize(self, voxel_features, view, projection, background_feature):
        # require background voxel (e.g. making it X*Y*Z + 1 voxel latents) to voxel features grid. When nvdiffrast rasterizes, the background voxel features will be used for pixels that do not hit any voxel, as they're given a -1 index.
        B, N, C = voxel_features.shape

        assert N == self.dim ** 3

        voxel_features = torch.cat(
                [voxel_features, repeat(background_feature, 'c -> b v c', b=B, v=1)], dim=1).float()

        ret = self.rasterize_voxel_peeling(
            self.ctx,
            self.vertices,
            self.faces,
            self.width,
            self.height,
            self.max_layers,
            view=view,
            projection=projection,
            antialiasing=False,
            use_optimized=self.use_optimized
        )
        face_ids = torch.stack(ret['face_id'], dim=0).to(dtype=torch.int) - 1  # l, (b t), h, w
        depth = torch.stack(ret['depth'], dim=0)  # l, (b t), h, w
        voxel_ids = (face_ids // 12) % voxel_features.shape[1]  # 12 faces per voxel, then modulo to get +ve voxel_id for background voxel
        # voxel_features: [batch*time, num_voxels+1, feat_dim]
        # voxel_ids: [layers, batch*time, height, width]
        layers, bt, h, w = voxel_ids.shape
        # Build batch index aligned with voxel_ids
        batch_idx = torch.arange(bt, device=voxel_ids.device)[None, :, None, None]  # (1, b, 1, 1)
        pixel_features = voxel_features[batch_idx, voxel_ids]
        pixel_features = rearrange(pixel_features, 'l b h w c -> l b h w c')
        return pixel_features, depth
