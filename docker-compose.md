

### 首次部署/重建

```bash
docker-compose down --remove-orphans
docker-compose up -d --build
docker-compose ps
```

### 启动/停止/重启

```bash
docker-compose start
docker-compose stop
docker-compose restart
```

### 仅重启后端

```bash
docker-compose up -d --build --no-deps backend
```

## 日常巡检

### 服务状态

```bash
docker-compose ps
```

### 日志查看

```bash
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs --tail 200 backend
docker-compose logs --tail 200 frontend
```
