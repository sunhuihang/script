#wget 
wget -c https://srtm.csi.cgiar.org/TIFF/srtm_12_03.zip -O ../test
wget -i file_list.txt
#下载ftp链接下所有数据
wget -c ftp://ftp.gscloud.cn/* --ftp-user "${name}" --ftp-password "${password}"
wget -c -r ftp://ftp.gscloud.cn/* --ftp-user "${name}" --ftp-password "${password}" #连该文件夹一起下载

#解压tar/gz
tar xvf xx.tar 或 tar xvf xx.tar.gz
tar xvf xx.tar -C ../dir #解压到dir文件夹

# 使用 gzip 压缩（生成 .tar.gz 或 .tgz 文件）
tar -czvf archive.tar.gz my_folder/
# 使用 bzip2 压缩（生成 .tar.bz2 文件，压缩率更高但速度更慢）
tar -cjvf archive.tar.bz2 my_folder/
# 使用 xz 压缩（生成 .tar.xz 文件，压缩率最高但速度最慢）
tar -cJvf archive.tar.xz my_folder/



#解压zip
unzip -o xx.zip -d dir #-o覆盖重名文件,-d指定解压dir目录下
#把文件夹压缩为zip
zip -r output.zip /path/to/directory
把指定文件压缩为zip
zip output.zip /path/to/file

#跨服务器传输，sshpass -p pwd123 是免互动输入密码
sshpass -p pwd123 scp -r qixiang@192.168.0.18:/home/qixiang/SHARE/us_data .
sshpass -p pwd123 rsync -avz qixiang@192.168.0.18:/home/qixiang/SHARE/us_data . #-a 为保持原始权限，v是显示详细，z是压缩传输

scp 指定端口 -P 10023

#在shell中用函数封装实现批量注释的作用
#把command1和2封装到dump函数中，但不调用，相当于注释掉了
dump(){
command1
command2
}
