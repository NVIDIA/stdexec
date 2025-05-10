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
	"-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT:STRING=Embedded" `
	"-DSTDEXEC_ENABLE_ASIO:BOOL=TRUE" `
	"-DSTDEXEC_ASIO_IMPLEMENTATION:STRING=boost" .
Invoke-NativeCommand cmake --build $BuildDirectory
Invoke-NativeCommand ctest --test-dir $BuildDirectory
