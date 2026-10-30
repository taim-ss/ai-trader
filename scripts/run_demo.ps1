param([switch]$Execute, [switch]$Approve, [switch]$Phone, [double]$BudgetEur = 200, [string]$Signal = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Python 3.12 must be installed before running the demo bot.' }
foreach ($variableName in @('TRADING212_DEMO_API_KEY', 'TRADING212_DEMO_API_SECRET')) {
    $credentialValue = [Environment]::GetEnvironmentVariable($variableName, 'User')
    if (-not [string]::IsNullOrWhiteSpace($credentialValue)) {
        Set-Item -Path "Env:$variableName" -Value $credentialValue
    }
}
$credentialValue = $null
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$runnerArguments = @((Join-Path $PSScriptRoot 'paper_rebalance.py'), '--budget-eur', $BudgetEur.ToString([Globalization.CultureInfo]::InvariantCulture))
if ($Signal) { $runnerArguments += @('--signal', (Resolve-Path -LiteralPath $Signal).Path) }
if ($Execute) { $runnerArguments += @('--execute', '--demo-only') }
if ($Approve) { $runnerArguments += @('--approve', '--demo-only') }
if ($Phone) { $runnerArguments += @('--phone', '--demo-only') }
Push-Location -LiteralPath $projectRoot
try { & $pythonPath @runnerArguments; $runnerExitCode = $LASTEXITCODE }
finally { Pop-Location }
exit $runnerExitCode
