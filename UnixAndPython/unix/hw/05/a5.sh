#!/usr/bin/env bash
cat $1 | xargs -n1 factor > $2
