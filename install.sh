wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin
mc alias set myminio http://localhost:9000 minio minio123
mc mb myminio/mybucket