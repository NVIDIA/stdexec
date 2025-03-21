.. =============================================================================
..  Copyright 2025 NVIDIA Corporation
.. 
..  Licensed under the Apache License, Version 2.0 (the "License");
..  you may not use this file except in compliance with the License.
..  You may obtain a copy of the License at
.. 
..      http://www.apache.org/licenses/LICENSE-2.0
.. 
..  Unless required by applicable law or agreed to in writing, software
..  distributed under the License is distributed on an "AS IS" BASIS,
..  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
..  See the License for the specific language governing permissions and
..  limitations under the License.
.. =============================================================================

Developer's Guide
=================

This section will eventually contain a guide for those interested in developing their own
asynchronous, sender-based algorithms and execution contexts.

Essential concepts
------------------

* Receivers
* Custom Algorithms
* Custom Schedulers
* Customizing ``stdexec``'s algorithms
    * Domains
    * Early algorithm customization
    * Late algorithm customization
