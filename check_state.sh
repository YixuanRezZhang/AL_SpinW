while true; 
do 
echo 'Test1'; cd Merged_Kpath_test1/; cat outinfo | grep 'TR' | tail -3; cat dict_candi | wc -l;  echo ' '; cd /work/projects/p0020541/Yixuan/SpinWave;
echo 'Test2'; cd Merged_Kpath_test2/; cat outinfo | grep 'TR' | tail -3; cat dict_candi | wc -l;  echo ' '; cd /work/projects/p0020541/Yixuan/SpinWave;
echo 'Test3'; cd Merged_Kpath_test3/; cat outinfo | grep 'TR' | tail -3; cat dict_candi | wc -l;  echo ' '; cd /work/projects/p0020541/Yixuan/SpinWave;
echo 'AP_Test1'; cd Allpara_Merged_Kpath_test1/; cat outinfo | grep 'TR' | tail -3; cat dict_candi | wc -l;  echo ' '; cd /work/projects/p0020541/Yixuan/SpinWave;
echo 'AP_Test2'; cd Allpara_Merged_Kpath_test2/; cat outinfo | grep 'TR' | tail -3; cat dict_candi | wc -l;  echo ' '; cd /work/projects/p0020541/Yixuan/SpinWave;
echo 'AP_Test3'; cd Allpara_Merged_Kpath_test3/; cat outinfo | grep 'TR' | tail -3; cat dict_candi | wc -l;  echo ' '; cd /work/projects/p0020541/Yixuan/SpinWave;
sleep 120s; 
done