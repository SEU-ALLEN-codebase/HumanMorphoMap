#!/bin/bash

[ $# -ne 1 ] && { echo "用法: $0 <目录路径>"; exit 1; }

find "$1" -type f -name "*.swc" -print0 | \
parallel -0 '
    filename={};
    base=$(basename "$filename" .swc);
    awk -v name="$base" '\''
        !/^#/ && $2 == 4 {found=1; exit} 
        END {if (found) print name}
    '\'' "$filename"
'

