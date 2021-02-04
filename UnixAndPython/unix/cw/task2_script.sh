#!/usr/bin/env bash

grep -Pe "^(\<[a-z]+)((\s*\/\>)|(.*\<\/[a-z]*\>))$"