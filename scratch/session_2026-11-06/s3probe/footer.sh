#!/bin/bash
# usage: footer.sh <s3 key> <local name>
E="--profile runpod --endpoint-url https://s3api-eu-ro-1.runpod.io --region eu-ro-1"
key="$1"; name="$2"
size=$(aws s3api head-object --bucket z7yvc40ykx --key "$key" $E --query ContentLength --output text)
aws s3api get-object --bucket z7yvc40ykx --key "$key" --range "bytes=$((size-8))-$((size-1))" "$name.tail8" $E >/dev/null
flen=$(/g/pureskill/envs/tabicl/Scripts/python.exe -I -c "import struct,sys;b=open(sys.argv[1],'rb').read();print(struct.unpack('<I',b[:4])[0], b[4:8])" "$name.tail8")
echo "$key size=$size footer_len_and_magic=$flen"
L=$(echo "$flen" | sed -E "s/^([0-9]+).*/\1/")
aws s3api get-object --bucket z7yvc40ykx --key "$key" --range "bytes=$((size-L-8))-$((size-1))" "$name.footer" $E >/dev/null
rm -f "$name.tail8"
