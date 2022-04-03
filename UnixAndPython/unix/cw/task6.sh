#!/usr/bin/env bash

"$(base64 $1)" | tee >/dev/null "$1"