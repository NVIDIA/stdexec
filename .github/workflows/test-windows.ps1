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

# Run the tests under Application Verifier to check for runtime failures like
# use-after-free.
Invoke-NativeCommand appverif /verify test.stdexec.exe
Invoke-NativeCommand appverif /verify test.exec.exe

Invoke-NativeCommand ctest --test-dir $BuildDirectory --output-on-failure --verbose --timeout 60

# Reset the Application Verifier settings for the test executables.
Invoke-NativeCommand appverif /n test.stdexec.exe
Invoke-NativeCommand appverif /n test.exec.exe
