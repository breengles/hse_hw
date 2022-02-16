#!/usr/bin/env bash


docker run --mount src=$(pwd)/artifacts,target=/advpy/hw2/artifacts,type=bind -it advpyhw2
