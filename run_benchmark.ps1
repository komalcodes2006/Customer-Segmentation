[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$sequentialExe = Join-Path $projectRoot 'bin\kmeans_sequential.exe'
$parallelExe = Join-Path $projectRoot 'bin\kmeans_parallel.exe'
$benchmarkDirectory = Join-Path $projectRoot 'data\processed\benchmarks'
$resultsDirectory = Join-Path $projectRoot 'results'
$resultsPath = Join-Path $resultsDirectory 'performance_results.csv'

$datasetSizes = @(10000, 50000, 100000, 500000, 1000000)
$threadCounts = @(1, 2, 4, 8)

function Assert-FileExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing $Description`: $Path"
    }
}

function Convert-ToInvariantDouble {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    return [double]::Parse(
        $Value.Trim(),
        [Globalization.CultureInfo]::InvariantCulture
    )
}

function Invoke-KMeans {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(Mandatory = $true)]
        [string]$DatasetPath,

        [Parameter(Mandatory = $true)]
        [string]$Implementation,

        [Parameter(Mandatory = $true)]
        [int]$DatasetSize,

        [Parameter(Mandatory = $true)]
        [int]$Threads
    )

    if ($Implementation -eq 'Parallel') {
        $env:OMP_NUM_THREADS = [string]$Threads
    }

    $rawOutput = (& $Executable $DatasetPath 2>&1 | Out-String)
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        throw "$Implementation benchmark failed for dataset size $DatasetSize (exit code $exitCode). Output:`n$rawOutput"
    }

    $executionMatch = [regex]::Match(
        $rawOutput,
        '(?im)^Execution time:\s*([0-9eE+\-.]+)\s+seconds\s*$'
    )
    $iterationsMatch = [regex]::Match(
        $rawOutput,
        '(?im)^Iterations:\s*([0-9]+)\s*$'
    )
    $inertiaMatch = [regex]::Match(
        $rawOutput,
        '(?im)^Inertia:\s*([0-9eE+\-.]+)\s*$'
    )

    if (-not $executionMatch.Success -or
        -not $iterationsMatch.Success -or
        -not $inertiaMatch.Success) {
        throw "Could not parse required benchmark metrics for $Implementation, dataset size $DatasetSize. Output:`n$rawOutput"
    }

    $clusterCounts = @('', '', '', '')
    for ($cluster = 0; $cluster -lt 4; $cluster++) {
        $clusterMatch = [regex]::Match(
            $rawOutput,
            "(?im)^Cluster\s+${cluster}:\s*([0-9]+)\s*$"
        )

        if ($clusterMatch.Success) {
            $clusterCounts[$cluster] = [int]$clusterMatch.Groups[1].Value
        }
    }

    return [pscustomobject]@{
        DatasetSize = $DatasetSize
        Implementation = $Implementation
        Threads = if ($Implementation -eq 'Sequential') { 1 } else { $Threads }
        ExecutionTimeSeconds = Convert-ToInvariantDouble $executionMatch.Groups[1].Value
        Iterations = [int]$iterationsMatch.Groups[1].Value
        Inertia = Convert-ToInvariantDouble $inertiaMatch.Groups[1].Value
        Cluster0 = $clusterCounts[0]
        Cluster1 = $clusterCounts[1]
        Cluster2 = $clusterCounts[2]
        Cluster3 = $clusterCounts[3]
        ClusterSizes = ($clusterCounts -join ';')
        Speedup = $null
        ParallelEfficiencyPercent = $null
        Output = $rawOutput.TrimEnd()
    }
}

Assert-FileExists $sequentialExe 'sequential executable'
Assert-FileExists $parallelExe 'parallel executable'

foreach ($size in $datasetSizes) {
    $datasetPath = Join-Path $benchmarkDirectory "customers_${size}.csv"
    Assert-FileExists $datasetPath "benchmark dataset for size $size"
}

if (-not (Test-Path -LiteralPath $resultsDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $resultsDirectory -Force | Out-Null
}

$results = New-Object System.Collections.Generic.List[object]

foreach ($size in $datasetSizes) {
    $datasetPath = Join-Path $benchmarkDirectory "customers_${size}.csv"

    Write-Host ''
    Write-Host '========================================'
    Write-Host ("Dataset: {0:N0} customers" -f $size)
    Write-Host '========================================'

    Write-Host 'Running Sequential...'
    $sequentialResult = Invoke-KMeans `
        -Executable $sequentialExe `
        -DatasetPath $datasetPath `
        -Implementation 'Sequential' `
        -DatasetSize $size `
        -Threads 1
    $results.Add($sequentialResult)

    foreach ($threads in $threadCounts) {
        $threadLabel = if ($threads -eq 1) { 'thread' } else { 'threads' }
        Write-Host ("Running Parallel - {0} {1}..." -f $threads, $threadLabel)

        $parallelResult = Invoke-KMeans `
            -Executable $parallelExe `
            -DatasetPath $datasetPath `
            -Implementation 'Parallel' `
            -DatasetSize $size `
            -Threads $threads

        $parallelResult.Speedup =
            $sequentialResult.ExecutionTimeSeconds / $parallelResult.ExecutionTimeSeconds
        $parallelResult.ParallelEfficiencyPercent =
            ($parallelResult.Speedup / $threads) * 100.0
        $results.Add($parallelResult)
    }
}

$results |
    Select-Object DatasetSize, Implementation, Threads, ExecutionTimeSeconds,
        Iterations, Inertia, Cluster0, Cluster1, Cluster2, Cluster3,
        ClusterSizes, Speedup, ParallelEfficiencyPercent, Output |
    Export-Csv -LiteralPath $resultsPath -NoTypeInformation -Encoding UTF8

Write-Host ''
Write-Host '========================================'
Write-Host 'BENCHMARK COMPLETE'
Write-Host '========================================'
Write-Host 'Results saved to:'
Write-Host 'results\performance_results.csv'
