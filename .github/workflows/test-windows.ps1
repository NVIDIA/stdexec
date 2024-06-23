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

Invoke-NativeCommand cmake -B $BuildDirectory -G Ninja "-DCMAKE_BUILD_TYPE=$Config" .
Invoke-NativeCommand cmake --build $BuildDirectory --parallel 1
Invoke-NativeCommand ctest --test-dir $BuildDirectory
