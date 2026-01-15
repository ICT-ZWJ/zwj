cd /root/zhuweijie/Chirality_in_Action/ssv2_dataset/

# 1. 合并两个分卷
cat 20bn-something-something-v2-00 20bn-something-something-v2-01 > combined.tar.gz

# 2. 解压
mkdir -p ssv2_videos
tar -xzvf combined.tar.gz -C ssv2_videos/

# 3. 查看结果
echo "解压完成！"
ls -lh ssv2_videos/
find ssv2_videos/ -type f -name "*.webm" | wc -l

# 4. 清理临时文件
rm combined.tar.gz