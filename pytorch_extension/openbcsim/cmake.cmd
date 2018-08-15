cmake ^
-E ^
env ^
LDFLAGS="-fuse-ld=lld" ^
cmake ^
. ^
-Bbuild ^
-GNinja ^
-DCMAKE_CXX_COMPILER:PATH="clang++.exe" ^
-DCMAKE_CXX_COMPILER_ID="Clang" ^
-DCMAKE_SYSTEM_NAME="Generic"
