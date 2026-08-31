<#
    .SYNOPSIS
        Simple video/audio downloader — a small standalone Windows tool.

    .DESCRIPTION
        One window: paste a link, pick video (MP4) or audio (MP3), press download.
        Nothing else. This file is the ONLY source of truth for both shipping
        formats — the portable ZIP and the single EXE both run this exact script,
        so there is no second copy to drift.

        It is deliberately self-contained: it does not import, read or touch
        anything from the SubsTranslator project it happens to live next to.

    .PARAMETER SelfTest
        Prints the yt-dlp argument list it would build, as JSON, and exits.
        No window, no download. This is what CI asserts on, because the argument
        list is the part most likely to be silently wrong.

    .NOTES
        Saved as UTF-8 WITH BOM on purpose: Windows PowerShell 5.1 reads a BOM-less
        file as ANSI, which turns every Hebrew string in here into garbage.
#>
[CmdletBinding()]
param(
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Paths and tools
# ---------------------------------------------------------------------------

function Get-ScriptRoot {
    # $PSScriptRoot is empty when the script is wrapped into an EXE, so fall back
    # to the running process's own folder.
    if ($PSScriptRoot) { return $PSScriptRoot }
    if ($MyInvocation.MyCommand.Path) { return (Split-Path -Parent $MyInvocation.MyCommand.Path) }
    return [System.IO.Path]::GetDirectoryName([System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName)
}

function Get-ToolLocations {
    <#
        Two shipping modes, one script:
          portable  - bin\ sits next to the script (the ZIP). Nothing to download.
          installed - nothing next to us (the EXE), so tools live in LOCALAPPDATA
                      and are fetched on first run.
    #>
    $root = Get-ScriptRoot
    $portableBin = [System.IO.Path]::Combine($root, 'bin')
    if (Test-Path ([System.IO.Path]::Combine($portableBin, 'yt-dlp.exe'))) {
        return [pscustomobject]@{
            Mode   = 'portable'
            BinDir = $portableBin
            YtDlp  = [System.IO.Path]::Combine($portableBin, 'yt-dlp.exe')
            Ffmpeg = [System.IO.Path]::Combine($portableBin, 'ffmpeg.exe')
        }
    }
    $appBin = [System.IO.Path]::Combine($env:LOCALAPPDATA, 'SimpleDownloader', 'bin')
    return [pscustomobject]@{
        Mode   = 'installed'
        BinDir = $appBin
        YtDlp  = [System.IO.Path]::Combine($appBin, 'yt-dlp.exe')
        Ffmpeg = [System.IO.Path]::Combine($appBin, 'ffmpeg.exe')
    }
}

function Get-DownloadsFolder {
    $downloads = [System.IO.Path]::Combine([Environment]::GetFolderPath('UserProfile'), 'Downloads')
    if (Test-Path $downloads) { return $downloads }
    return [Environment]::GetFolderPath('MyDocuments')
}

# ---------------------------------------------------------------------------
# The argument list — the part worth testing
# ---------------------------------------------------------------------------

function Get-DownloadArguments {
    param(
        [Parameter(Mandatory)][string]$Url,
        [Parameter(Mandatory)][ValidateSet('mp4', 'mp3')][string]$MediaFormat,
        [Parameter(Mandatory)][string]$OutputDir,
        [string]$FfmpegPath,
        [string]$ResultPath
    )

    $arguments = @(
        '--no-playlist'          # a link that carries a playlist gives one video
        '--newline'              # progress on its own line, so it can be parsed
        '--no-mtime'             # keep the download time, not YouTube's upload time
        '--retries', '5'
        '--fragment-retries', '5'
        '--socket-timeout', '30'
        # Path.Combine, not Join-Path: Join-Path goes through PowerShell's
        # provider layer, which resolves drives and treats [ ] as wildcards. This is
        # a string handed to another program, so keep it away from all of that.
        '-o', ([System.IO.Path]::Combine($OutputDir, '%(title).100B.%(ext)s'))
    )

    if ($ResultPath) {
        # Ask yt-dlp where it actually put the file instead of guessing from the
        # template: the extension changes during post-processing.
        $arguments += @('--print-to-file', 'after_move:filepath', $ResultPath)
    }
    if ($FfmpegPath) {
        $arguments += @('--ffmpeg-location', $FfmpegPath)
    }

    if ($MediaFormat -eq 'mp3') {
        $arguments += @(
            '-f', 'bestaudio/best'
            '-x'
            '--audio-format', 'mp3'
            '--audio-quality', '192'
        )
    }
    else {
        # avc1 + m4a first: the combination Windows plays without extra codecs.
        $arguments += @(
            '-f', 'bestvideo[height<=1080][vcodec^=avc1]+bestaudio[acodec^=mp4a]/bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best'
            '--merge-output-format', 'mp4'
        )
    }

    $arguments += $Url
    return $arguments
}

function ConvertTo-ArgumentString {
    param([string[]]$Arguments)
    # Built by hand rather than handed to -ArgumentList as an array: PowerShell's
    # own array joining does not quote reliably, and one unquoted path with a
    # space in the user name turns into two arguments.
    ($Arguments | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join ' '
}

# ---------------------------------------------------------------------------
# Self test — no GUI
# ---------------------------------------------------------------------------

if ($SelfTest) {
    $report = [ordered]@{}
    foreach ($mediaFormat in @('mp4', 'mp3')) {
        $arguments = Get-DownloadArguments -Url 'https://example.com/watch?v=abc' `
            -MediaFormat $mediaFormat `
            -OutputDir ([System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), 'Test User')) `
            -FfmpegPath ([System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), 'ffmpeg.exe')) `
            -ResultPath ([System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), 'result.txt'))
        $report[$mediaFormat] = [ordered]@{
            Arguments = $arguments
            Line      = ConvertTo-ArgumentString $arguments
        }
    }
    $report | ConvertTo-Json -Depth 5
    exit 0
}

# ---------------------------------------------------------------------------
# First run: fetch yt-dlp and ffmpeg (EXE mode only)
# ---------------------------------------------------------------------------

$script:YtDlpUrl = 'https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe'
$script:FfmpegUrl = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'

function Update-YtDlp {
    <#
        yt-dlp is the one moving part: YouTube changes something every few weeks
        and the fix always ships as a new yt-dlp release. `-U` is the official
        binary's built-in self-update. ffmpeg is deliberately NOT updated - it
        has no YouTube coupling and never needs to move.

        Failure here must never block a download attempt: the current version
        may still work, and if it does not, the download error will say so.
        Bounded wait so a hung network cannot freeze the window forever.
    #>
    param([Parameter(Mandatory)]$Tools)
    try {
        $process = Start-Process -FilePath $Tools.YtDlp -ArgumentList '-U' `
            -WindowStyle Hidden -PassThru
        if (-not $process.WaitForExit(60000)) {
            $process.Kill()
            return $false
        }
        return ($process.ExitCode -eq 0)
    }
    catch { return $false }
}

function Install-Tools {
    param([Parameter(Mandatory)]$Tools)

    New-Item -ItemType Directory -Force -Path $Tools.BinDir | Out-Null
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

    if (-not (Test-Path $Tools.YtDlp)) {
        Invoke-WebRequest -Uri $script:YtDlpUrl -OutFile $Tools.YtDlp -UseBasicParsing
    }

    if (-not (Test-Path $Tools.Ffmpeg)) {
        $zipPath = [System.IO.Path]::Combine($env:TEMP, 'ffmpeg-win64.zip')
        $extractDir = [System.IO.Path]::Combine($env:TEMP, 'ffmpeg-win64')
        Invoke-WebRequest -Uri $script:FfmpegUrl -OutFile $zipPath -UseBasicParsing
        if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force }
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
        $found = Get-ChildItem -Path $extractDir -Filter 'ffmpeg.exe' -Recurse | Select-Object -First 1
        if (-not $found) { throw 'ffmpeg.exe was not found inside the downloaded archive' }
        Copy-Item $found.FullName $Tools.Ffmpeg -Force
        Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
        Remove-Item $extractDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$script:Tools = Get-ToolLocations
$script:OutputDir = Get-DownloadsFolder
$script:Process = $null
$script:StdOutPath = $null
$script:StdErrPath = $null
$script:ResultPath = $null
$script:LastFile = $null

$form = New-Object System.Windows.Forms.Form
$form.Text = 'הורדת סרטונים'
$form.Size = New-Object System.Drawing.Size(560, 340)
$form.StartPosition = 'CenterScreen'
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$form.RightToLeft = 'Yes'
$form.RightToLeftLayout = $true
$form.Font = New-Object System.Drawing.Font('Segoe UI', 10)

$labelUrl = New-Object System.Windows.Forms.Label
$labelUrl.Text = 'הדביקי כאן קישור לסרטון:'
$labelUrl.Location = New-Object System.Drawing.Point(20, 20)
$labelUrl.Size = New-Object System.Drawing.Size(500, 24)
$form.Controls.Add($labelUrl)

$textUrl = New-Object System.Windows.Forms.TextBox
$textUrl.Location = New-Object System.Drawing.Point(20, 48)
$textUrl.Size = New-Object System.Drawing.Size(500, 28)
$textUrl.RightToLeft = 'No'   # a URL is left-to-right even on a Hebrew window
$form.Controls.Add($textUrl)

$groupFormat = New-Object System.Windows.Forms.GroupBox
$groupFormat.Text = 'מה להוריד'
$groupFormat.Location = New-Object System.Drawing.Point(20, 90)
$groupFormat.Size = New-Object System.Drawing.Size(500, 62)
$form.Controls.Add($groupFormat)

$radioVideo = New-Object System.Windows.Forms.RadioButton
$radioVideo.Text = 'וידאו (MP4)'
$radioVideo.Location = New-Object System.Drawing.Point(20, 24)
$radioVideo.Size = New-Object System.Drawing.Size(180, 26)
$radioVideo.Checked = $true
$groupFormat.Controls.Add($radioVideo)

$radioAudio = New-Object System.Windows.Forms.RadioButton
$radioAudio.Text = 'אודיו בלבד (MP3)'
$radioAudio.Location = New-Object System.Drawing.Point(220, 24)
$radioAudio.Size = New-Object System.Drawing.Size(220, 26)
$groupFormat.Controls.Add($radioAudio)

$buttonDownload = New-Object System.Windows.Forms.Button
$buttonDownload.Text = 'הורד'
$buttonDownload.Location = New-Object System.Drawing.Point(20, 164)
$buttonDownload.Size = New-Object System.Drawing.Size(140, 38)
$form.Controls.Add($buttonDownload)

$buttonOpen = New-Object System.Windows.Forms.Button
$buttonOpen.Text = 'פתח את התיקייה'
$buttonOpen.Location = New-Object System.Drawing.Point(174, 164)
$buttonOpen.Size = New-Object System.Drawing.Size(160, 38)
$buttonOpen.Enabled = $false
$form.Controls.Add($buttonOpen)

$progress = New-Object System.Windows.Forms.ProgressBar
$progress.Location = New-Object System.Drawing.Point(20, 216)
$progress.Size = New-Object System.Drawing.Size(500, 22)
$form.Controls.Add($progress)

$labelStatus = New-Object System.Windows.Forms.Label
$labelStatus.Text = ''
$labelStatus.Location = New-Object System.Drawing.Point(20, 244)
$labelStatus.Size = New-Object System.Drawing.Size(500, 48)
$form.Controls.Add($labelStatus)

function Set-Status {
    param([string]$Text)
    $labelStatus.Text = $Text
    $labelStatus.Refresh()
}

function Set-Busy {
    param([bool]$Busy)
    $buttonDownload.Enabled = -not $Busy
    $textUrl.Enabled = -not $Busy
    $radioVideo.Enabled = -not $Busy
    $radioAudio.Enabled = -not $Busy
}

# --- the download itself -----------------------------------------------------

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 400

function Read-SharedText {
    param([string]$Path)
    # yt-dlp still holds the file open, so a plain Get-Content would fail.
    if (-not (Test-Path $Path)) { return '' }
    try {
        $stream = [System.IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite')
        try {
            $reader = New-Object System.IO.StreamReader($stream)
            return $reader.ReadToEnd()
        }
        finally { $stream.Dispose() }
    }
    catch { return '' }
}

function Stop-Tracking {
    $timer.Stop()
    Set-Busy $false
}

$timer.Add_Tick({
    $output = Read-SharedText $script:StdOutPath
    # Not $matches: that is an automatic PowerShell variable and clobbering it
    # inside an event handler is a good way to break something unrelated.
    $percentHits = [regex]::Matches($output, '(\d{1,3}(?:\.\d)?)%')
    if ($percentHits.Count -gt 0) {
        $percent = [int][double]$percentHits[$percentHits.Count - 1].Groups[1].Value
        if ($percent -ge 0 -and $percent -le 100) { $progress.Value = $percent }
    }
    if ($output -match '\[ExtractAudio\]|\[Merger\]') {
        Set-Status 'ממיר את הקובץ...'
    }

    if ($script:Process -and $script:Process.HasExited) {
        Stop-Tracking
        if ($script:Process.ExitCode -eq 0) {
            $progress.Value = 100
            $script:LastFile = (((Read-SharedText $script:ResultPath).Trim() -split "`r?`n") | Where-Object { $_ } | Select-Object -Last 1)
            $buttonOpen.Enabled = $true
            $name = if ($script:LastFile) { Split-Path -Leaf $script:LastFile } else { 'הקובץ' }
            Set-Status "הסתיים! $name נשמר בתיקיית ההורדות."
        }
        else {
            $progress.Value = 0
            $errorText = (Read-SharedText $script:StdErrPath).Trim()
            $lastLine = ($errorText -split "`r?`n" | Where-Object { $_ } | Select-Object -Last 1)
            Set-Status 'ההורדה נכשלה.'
            $message = "ההורדה נכשלה.`n`nייתכן שהקישור שגוי, שהסרטון פרטי, או שאין חיבור לאינטרנט."
            if ($lastLine) { $message += "`n`nההודעה מהכלי:`n$lastLine" }
            [System.Windows.Forms.MessageBox]::Show($message, 'שגיאה', 'OK', 'Error') | Out-Null
        }
    }
})

$buttonDownload.Add_Click({
    $url = $textUrl.Text.Trim()
    if (-not $url) {
        [System.Windows.Forms.MessageBox]::Show('צריך להדביק קישור קודם.', 'רגע', 'OK', 'Information') | Out-Null
        return
    }
    if ($url -notmatch '^https?://') {
        [System.Windows.Forms.MessageBox]::Show('הקישור צריך להתחיל ב-http או https.', 'קישור לא תקין', 'OK', 'Warning') | Out-Null
        return
    }

    $mediaFormat = if ($radioAudio.Checked) { 'mp3' } else { 'mp4' }

    $stamp = [Guid]::NewGuid().ToString('N')
    $script:StdOutPath = [System.IO.Path]::Combine($env:TEMP, "dl-$stamp.out")
    $script:StdErrPath = [System.IO.Path]::Combine($env:TEMP, "dl-$stamp.err")
    $script:ResultPath = [System.IO.Path]::Combine($env:TEMP, "dl-$stamp.path")
    $script:LastFile = $null
    $buttonOpen.Enabled = $false
    $progress.Value = 0

    $arguments = Get-DownloadArguments -Url $url -MediaFormat $mediaFormat `
        -OutputDir $script:OutputDir -FfmpegPath $script:Tools.Ffmpeg -ResultPath $script:ResultPath

    Set-Busy $true
    Set-Status 'מוריד...'
    try {
        $script:Process = Start-Process -FilePath $script:Tools.YtDlp `
            -ArgumentList (ConvertTo-ArgumentString $arguments) `
            -RedirectStandardOutput $script:StdOutPath `
            -RedirectStandardError $script:StdErrPath `
            -WindowStyle Hidden -PassThru
        $timer.Start()
    }
    catch {
        Stop-Tracking
        Set-Status 'לא הצלחתי להפעיל את כלי ההורדה.'
        [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, 'שגיאה', 'OK', 'Error') | Out-Null
    }
})

$buttonOpen.Add_Click({
    if ($script:LastFile -and (Test-Path $script:LastFile)) {
        Start-Process 'explorer.exe' -ArgumentList ('/select,"{0}"' -f $script:LastFile)
    }
    else {
        Start-Process 'explorer.exe' -ArgumentList $script:OutputDir
    }
})

$form.Add_Shown({
    $form.Activate()
    if (-not (Test-Path $script:Tools.YtDlp) -or -not (Test-Path $script:Tools.Ffmpeg)) {
        Set-Busy $true
        Set-Status 'הפעלה ראשונה: מוריד רכיבים (כ-100MB). זה קורה פעם אחת בלבד...'
        try {
            Install-Tools -Tools $script:Tools
            Set-Status 'מוכן. הדביקי קישור ולחצי הורד.'
        }
        catch {
            Set-Status 'לא הצלחתי להוריד את הרכיבים.'
            [System.Windows.Forms.MessageBox]::Show(
                "לא הצלחתי להוריד את הרכיבים הנחוצים.`n`n$($_.Exception.Message)",
                'שגיאה', 'OK', 'Error') | Out-Null
        }
        Set-Busy $false
    }
    else {
        Set-Busy $true
        Set-Status 'בודק אם יש עדכון לכלי ההורדה...'
        Update-YtDlp -Tools $script:Tools | Out-Null
        Set-Busy $false
        Set-Status 'מוכן. הדביקי קישור ולחצי הורד.'
    }
})

[void]$form.ShowDialog()
