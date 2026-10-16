from toolpath_utils import (
    get_single_shell_contours,
    get_single_shell_contours_field_projection,
    del_non_enclosed_subcontours,
    compute_subcontour_centers,
    compute_approximate_normals,
    compute_approximate_normals_pcd,
)
from vis import *
from slice_meshing import *


def single_shell_toolpath_gen_mesh(udf_decoder, slice_decoder, data_paras, toolpath_paras, save_path):
    """
    single shell surface iso-contours extraction from triangle soups
    if mesh is given, use stable iso-contouring with cuda module.
    
    contours planning using reeb-graph can refer to INF-3DP.
    not provided here.
    """

    N = toolpath_paras.get('resolution')
    slice_num = toolpath_paras.get('slice_num')
    mesh_fn = data_paras.get('mesh_path')   # use out shell
    connect_broken=toolpath_paras.get('connect_broken', False)
    del_broken_segments=toolpath_paras.get('del_broken_segments', False)

    # ----------- extract iso-contours w/o normals (final pseudo-normal appro)
    init_contours, avg_edge_len = \
        get_single_shell_contours(slice_decoder, slice_num, mesh_fn, len_lim=0, 
                                connect_broken=connect_broken, discard_fisrt_num=1)
    
    if del_broken_segments:
        init_contours = del_non_enclosed_subcontours(init_contours)

    visualize_merged_iso_contours_mul(init_contours, start=None, end=None, line_width=7, random_color=True)

    # show subcontour centers
    centers = compute_subcontour_centers(init_contours)
    visualize_subcontour_centers(centers, mesh_fn=mesh_fn, point_size=15)

    init_contour_normals = compute_approximate_normals(init_contours, mesh_fn=mesh_fn)
    vis_contours_normals(init_contours, init_contour_normals, start=None, end=None)




def single_shell_toolpath_gen_field(udf_decoder, slice_decoder, data_paras, toolpath_paras, save_path):
    """
    single shell surface iso-contours extraction from field projection with UDF/GWN guidance field.
    applicable even when mesh is not given, but less robust than mesh iso-contouring

    stage 1: iso-contours tracing with UDF/GWN guidance field projection
    stage 2: parallel neighborhood contouring (connecting points into subcontours)

    The contours connecting relies on neighborhood search, 
    less robust than given-mesh iso-contouring which offers deterministic connection.
    P.s. UDF cannot be guaranteed to be same as original geometry.

    contours planning using reeb-graph can refer to INF-3DP.
    not provided here.
    """

    N = toolpath_paras.get('resolution')
    slice_num = toolpath_paras.get('slice_num')
    pcd_fn = data_paras.get('point_cloud_path')   
    connect_broken=toolpath_paras.get('connect_broken', False)
    del_broken_segments=toolpath_paras.get('del_broken_segments', False)

    # ----------- extract iso-contours w/o normals (final nearest appro)
    # len_lim to remove small segments
    init_contours, avg_edge_len = \
        get_single_shell_contours_field_projection(udf_decoder, slice_decoder, slice_num, pcd_fn, len_lim=10, 
                                                   connect_broken=connect_broken, discard_fisrt_num=1,
                                                   resolution=N, max_batch_points=64**3,
                                                   tracing_max_iter=10)
    
    if del_broken_segments:
        init_contours = del_non_enclosed_subcontours(init_contours)

    visualize_merged_iso_contours_mul(init_contours, start=None, end=None, line_width=7, random_color=True)

    # show subcontour centers
    centers = compute_subcontour_centers(init_contours)
    visualize_subcontour_centers(centers, mesh_fn=None, point_size=15)

    init_contour_normals = compute_approximate_normals_pcd(init_contours, pcd_fn=pcd_fn)
    vis_contours_normals(init_contours, init_contour_normals, start=None, end=None)


    


