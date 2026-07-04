#!/bin/bash
set -e

echo "🚀 VTON-TRYON 部署到 Replicate"
echo ""

# Check prerequisites
command -v cog >/dev/null 2>&1 || {
    echo "❌ 需要安装 cog CLI:"
    echo "   sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
    echo "   sudo chmod +x /usr/local/bin/cog"
    exit 1
}

command -v docker >/dev/null 2>&1 || {
    echo "❌ 需要安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
}

# Login
echo "🔐 Replicate 登录..."
cog login --token "${REPLICATE_API_TOKEN:-$(read -p 'Replicate API Token: ' tok && echo $tok)}"

# Push (builds Docker image + pushes to Replicate)
echo "🔨 构建并推送（首次约 15-20 分钟，含模型下载）..."
echo "   模型权重约 12GB，会打包进 Docker 镜像"
echo ""
cog push r8.im/songdiandong/vton-tryon

echo ""
echo "✅ 部署完成！"
echo "   模型页面: https://replicate.com/songdiandong/vton-tryon"
echo ""
echo "测试调用:"
echo '   python3 -c "'
echo '   import replicate'
echo '   output = replicate.run('
echo '       \"songdiandong/vton-tryon\",'
echo '       input={'
echo '           \"person_image\": open(\"person.jpg\", \"rb\"),'
echo '           \"garment_image\": open(\"garment.jpg\", \"rb\"),'
echo '           \"clothing_type\": \"tshirt\",'
echo '           \"garment_description\": \"a white t-shirt\",'
echo '       }'
echo '   )'
echo '   print(output)"'
