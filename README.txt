#######
3D EDSR
#######

训练：
nrrd格式的数据集放到original_dataset中。
python dataprocess.py
python 3D-EDSR.py

测试：
python 3D-EDSR-test.py  test_dataset/test.nrrd  result.nrrd 
test_dataset是测试数据文件夹，test.nrrd是输入数据名称，result.nrrd是结果保存名称，保存在result文件夹中。

测试结果下采样：
cd result
python up4_to_original.py  input.nrrd  output.nrrd
input.nrrd是输入数据名称，output.nrrd是结果保存名称，保存在当前文件夹中。

测试结果冠状面、横断面、矢状面三个方向分帧：
python nrrdto3Dpngs.py  result/input.nrrd
result是result文件夹，input.nrrd是输入数据名称，结果保存在result_pngs中。
