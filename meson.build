project('stdexec', 'cpp', version: '0.11.0', license: 'Apache2')

stdexec_dep = declare_dependency(
    include_directories: include_directories('include'),
)

if not meson.is_subproject()

    install_subdir(
        'include/exec',
        install_dir: 'include',
        strip_directory: false,
    )
    install_subdir(
        'include/stdexec',
        install_dir: 'include',
        strip_directory: false,
    )

    pkgc = import('pkgconfig')
    pkgc.generate(
        name: 'stdexec',
        version: meson.project_version(),
        description: 'stdexec: experimental P2300 implementation',
    )
endif
