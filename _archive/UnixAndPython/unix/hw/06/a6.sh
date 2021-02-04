#!/usr/bin/env bash
find . -maxdepth 1 -user $(whoami) | grep "*.txt" | wc -l