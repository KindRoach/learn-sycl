# Build

## Linux

```bash
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -GNinja -S . -B build
cmake --build build
```

## Windows

```bash
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -GNinja -S . -B build
cmake --build build
```
