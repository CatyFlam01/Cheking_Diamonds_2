#!/bin/bash
# Мониторинг ресурсов Docker контейнера
# Автор: [Ваше Имя]

echo "=============================================="
echo "MONITORING DOCKER RESOURCES"
echo "=============================================="
echo ""

echo "CONTAINER STATUS:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

echo "RESOURCE USAGE:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
echo ""

echo "IMAGE SIZES:"
docker images diamonds* --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
echo ""

echo "LAST LOGS:"
docker logs diamonds-api --tail 10
echo ""

echo "HEALTH CHECK:"
curl -s http://localhost:8000/health | python -m json.tool
echo ""

echo "=============================================="
echo "MONITORING COMPLETE"
echo "=============================================="
