# Shell

```bash
# 遍历数组元素
items=(
file1
file2
file2
)
#items=("file1" "file2" "file2")
for item in "${items[@]}"; do
    echo "is $item"
done
```

