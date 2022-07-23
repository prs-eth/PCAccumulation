import numpy as np
import numba

@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(
        points,
        voxel_size,
        coors_range,
        num_points_per_voxel,
        coor_to_voxelidx,
        coors,
        max_voxels=20000
        ):
    """
    Args:
        points:                 [N, 4], [x,y,z, t]
        voxel_size:             [3]
        coors_range:            [6] range of coordinates
        num_point_per_voxel:    [max_voxel], save the number of points in each voxel
        coor_to_voxelidx:       [nz, ny, nx, nt] mark the map from each occupied discretized voxel to its index
        voxels:                 [max_voxel, max_points, 4]; zeror padded
        coors:                  [max_voxel, 4] save the valid occupied voxel [z,y,z,t]
    
    Return:
        voxel_num: number of valid voxels
    """
    N = points.shape[0]
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(4,), dtype=np.int32)
    voxel_num = 0
    
    ndim = 3
    ndim_minus_1 = ndim - 1
    point_to_voxel_map = -1 * np.ones((N,1)).astype(np.int32)
    
    for i in range(N):
        failed = False
        
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c # zyx
        coor[3] = int(points[i, -1])
        
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2], coor[3]]
        if voxelidx == -1:  # the voxel is not filled/marked yet
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2], coor[3]] = voxelidx
            coors[voxelidx] = coor

        num_points_per_voxel[voxelidx] += 1
        point_to_voxel_map[i] = voxelidx
    return voxel_num, point_to_voxel_map


def points_to_voxel(
        points, 
        voxel_size, 
        coors_range, 
        n_sweeps,
        max_voxels=20000, 
):
    """convert points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud)
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points. ndim = 4
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1] + (n_sweeps,)   # [nz, ny, nx, nt]
    
    
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    coors = np.zeros(shape=(max_voxels, 4), dtype=np.int32)

    voxel_num, point_to_voxel_map = _points_to_voxel_reverse_kernel(
            points,
            voxel_size,
            coors_range,
            num_points_per_voxel,
            coor_to_voxelidx,
            coors,
            max_voxels,
        )
    coors = coors[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    
    return coors, num_points_per_voxel, point_to_voxel_map


class Voxelization(object):
    def __init__(self, cfg):
        voxel_size = cfg['voxel_size']
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        range = cfg['range']
        self.point_cloud_range = np.array(range, dtype=np.float32)
        self.n_sweeps=cfg['n_sweeps']
        grid_size = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        self.grid_size = grid_size
        
        self.max_voxels = self.grid_size[0] * self.grid_size[1] * self.grid_size[2] * self.n_sweeps

                
    def __call__(self, points):
        """Apply voxelisation in 4D space (x, y, z, t)
        Args:
            points (np.array): [N, 4]

        Returns:
            [type]: [description]
        """
        grid_size = np.hstack((self.grid_size, np.array([self.n_sweeps])))
        coordinates, num_points, point_to_voxel_map = points_to_voxel(points=points,
                                                          voxel_size=self.voxel_size,
                                                          coors_range=self.point_cloud_range,
                                                          n_sweeps=self.n_sweeps,
                                                          max_voxels=self.max_voxels
                                                          )
        num_voxels = np.array([coordinates.shape[0]], dtype=np.int64)
        
        results = {
            'coordinates': coordinates, #[M, 4],  zyxt
            'num_voxels': num_voxels, 
            'shape': grid_size,  #[nx, ny, nz, t],
            'point_to_voxel_map': point_to_voxel_map
        }
        return results