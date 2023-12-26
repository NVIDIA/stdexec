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

$VSVersion = 2022
$VSEdition = 'Enterprise'
$Architecture = 'x64'

Push-Location "C:/Program Files/Microsoft Visual Studio/$VSVersion/$VSEdition/VC/Auxiliary/Build"
$VCVersion = Get-Content 'Microsoft.VCToolsVersion.default.txt'
cmd /c "vcvarsall.bat $Architecture -vcvars_ver=$VCVersion > nul & set" | ForEach-Object {
	if ($_ -match '^(.+?)=(.*)') {
		Set-Item -Force -Path "ENV:$($matches[1])" -Value $matches[2]
	}
}
Pop-Location

if ($Compiler -ne "cl") {
	$ENV:CXX=$Compiler
}

if (Test-Path -PathType Container $BuildDirectory) {
	Remove-Item -Recurse $BuildDirectory | Out-Null
}
New-Item -ItemType Directory $BuildDirectory | Out-Null

Invoke-NativeCommand cmake -B $BuildDirectory -G Ninja "-DCMAKE_BUILD_TYPE=$Config" .
Invoke-NativeCommand cmake --build $BuildDirectory
Invoke-NativeCommand ctest --test-dir $BuildDirectory
