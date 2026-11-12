param(
	[string]$BuildDirectory="build",
	[string]$Compiler="cl",
	[string]$Config="Debug"
)

function Invoke-NativeCommand($Command) {
	& $Command $Args

	if (!$?) {
		throw "${Command}: $LastExitCode"
	}
}

Set-Location -Path "$PSScriptRoot/../.." -PassThru

if (Test-Path -PathType Container $BuildDirectory) {
	Remove-Item -Recurse $BuildDirectory | Out-Null
}
New-Item -ItemType Directory $BuildDirectory | Out-Null

Invoke-NativeCommand cmake -B $BuildDirectory -G Ninja `
	"-DCMAKE_BUILD_TYPE=$Config" `
	"-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON" `
	"-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT:STRING=Embedded" `
	"-DCMAKE_CXX_FLAGS:STRING=/fsanitize=address /EHsc" `
	"-DSTDEXEC_ENABLE_ASIO:BOOL=TRUE" `
	"-DSTDEXEC_ASIO_IMPLEMENTATION:STRING=boost" `
	"-DSTDEXEC_BUILD_TESTS:BOOL=TRUE" .
Invoke-NativeCommand cmake --build $BuildDirectory

# Verify that public headers are self-contained, same as the Linux CI job.
# This is the only job in CI where WIN32 is true, so it's the only place
# exec/windows/filetime_clock.hpp and exec/windows/windows_thread_pool.hpp
# (SKIP_LINTING'd everywhere else per CMakeLists.txt) actually get compiled
# and checked. Gated to Debug so it doesn't run twice per compiler/toolset
# pairing, matching the "run once" approach on the Linux side.
if ($Config -eq "Debug") {
	Invoke-NativeCommand cmake --build $BuildDirectory --target all_verify_interface_header_sets
}

Invoke-NativeCommand ctest --test-dir $BuildDirectory --output-on-failure --verbose --timeout 60
