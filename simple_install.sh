#! /bin/sh -e

[ -f CODE_OF_CONDUCT.md ] || { echo "Please run $0 in root directory of stdexec"; exit 1; }
[ -d "$1" ] || { echo "First argument should be a directory"; exit 1; }

# exclude asio sub directory
find include/exec -path "include/exec/asio" -prune -o -type f -exec install -D -m 0644 {} "$1"/{} \;
find include/stdexec -type f -exec install -D -m 0644 {} "$1"/{} \;

cat << EOF > "$1"/include/stdexec_version_config.hpp
/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#define STDEXEC_VERSION_MAJOR 0
#define STDEXEC_VERSION_MINOR 11
#define STDEXEC_VERSION_PATCH 0
EOF
