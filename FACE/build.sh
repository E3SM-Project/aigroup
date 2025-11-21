#!/usr/bin/env bash

source load_env.sh

clean_flag=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --clean)
            clean_flag=true
            ;;
    esac
done

main() {

    mkdir build && cd build

    cmake -DCMAKE_INSTALL_PREFIX=".." ..
    cmake --build .
    cmake --install .

    cd ..
}

clean() {
    rm -rf build
}

if $clean_flag; then
    clean
fi

main


