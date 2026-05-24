#!/bin/bash
echo "Formatting all C++ files..."
find ./cpp ./benchmark -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) \
  -exec clang-format -i {} \; \
  -print
echo "✅ Done"