#!/bin/sh

dir_path="/workspace/task/valid_data_n_increase/script/one_pig_tracking_result/*"
dirs=`find $dir_path -maxdepth 0 -type d`

for dir in $dirs;
do
    echo $dir
    ffmpeg -y -framerate 6 -i $dir/bbox_tracking/%05d.jpg -vcodec libx264 -pix_fmt yuv420p -r 6 $dir/bbox_tracking/out.mp4
    ffmpeg -y -framerate 6 -i $dir/graph_sre/%05d.jpg     -vcodec libx264 -pix_fmt yuv420p -r 6 $dir/graph_sre/out.mp4
    ffmpeg -y -framerate 6 -i $dir/graph_sre_v2/%05d.jpg  -vcodec libx264 -pix_fmt yuv420p -r 6 $dir/graph_sre_v2/out.mp4
    ffmpeg -y -framerate 6 -i $dir/graph_araya/%05d.jpg   -vcodec libx264 -pix_fmt yuv420p -r 6 $dir/graph_araya/out.mp4
    ffmpeg -y -i $dir/bbox_tracking/out.mp4 -s 640x480 $dir/bbox_tracking/resize.mp4
    ffmpeg -y -i $dir/graph_sre/out.mp4     -s 640x480 $dir/graph_sre/resize.mp4
    ffmpeg -y -i $dir/graph_sre_v2/out.mp4  -s 640x480 $dir/graph_sre_v2/resize.mp4
    ffmpeg -y -i $dir/graph_araya/out.mp4   -s 640x480 $dir/graph_araya/resize.mp4

    ffmpeg -y -i $dir/bbox_tracking/resize.mp4 -i $dir/graph_sre/resize.mp4 -i $dir/graph_sre_v2/resize.mp4 -i $dir/graph_araya/resize.mp4 -filter_complex  "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map "[v]" $dir/output.mp4

    rm $dir/bbox_tracking/out.mp4
    rm $dir/graph_sre/out.mp4
    rm $dir/graph_sre_v2/out.mp4
    rm $dir/graph_araya/out.mp4
    rm $dir/bbox_tracking/resize.mp4
    rm $dir/graph_sre/resize.mp4
    rm $dir/graph_sre_v2/resize.mp4
    rm $dir/graph_araya/resize.mp4
done
