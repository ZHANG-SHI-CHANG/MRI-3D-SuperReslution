#######
3D EDSR
#######

ѵ����
nrrd��ʽ�����ݼ��ŵ�original_dataset�С�
python dataprocess.py
python 3D-EDSR.py

���ԣ�
python 3D-EDSR-test.py  test_dataset/test.nrrd  result.nrrd 
test_dataset�ǲ��������ļ��У�test.nrrd�������������ƣ�result.nrrd�ǽ���������ƣ�������result�ļ����С�

���Խ���²�����
cd result
python up4_to_original.py  input.nrrd  output.nrrd
input.nrrd�������������ƣ�output.nrrd�ǽ���������ƣ������ڵ�ǰ�ļ����С�

���Խ����״�桢����桢ʸ״�����������֡��
python nrrdto3Dpngs.py  result/input.nrrd
result��result�ļ��У�input.nrrd�������������ƣ����������result_pngs�С�
