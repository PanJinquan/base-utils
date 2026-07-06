# docker-compose 常用命令

### 首次部署/重建

```bash
docker compose start   # 启动服务
docker compose stop    # 停止服务
docker compose restart # 重启服务
docker compose down    # 删除所有容器
#docker compose down --remove-orphans # 停止并删除当前定义的容器
# up启动容器，-d后台运行，--build容器在后台启动，终端不会被占用，强制重新构建镜像
docker compose up -d --build
# 查看服务状态
docker compose ps
# 查看后端日志
docker compose logs -f backend
# 查看前端日志
docker compose logs -f frontend
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs --tail 200 backend
docker compose logs --tail 200 frontend
```

### 启动/停止/重启

```bash
docker compose start
docker compose stop
docker compose restart
```

### 仅重启后端

```bash
docker-compose up -d --build --no-deps backend
```

