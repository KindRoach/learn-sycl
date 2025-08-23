#!/bin/bash

vtune-backend \
    --no-https \
    --web-port 8080 \
    --enable-server-profiling
