# Define the path to your PEM file with private key for the certificate
$pemFilePath = ".\chatbotpoc.pem"

# Read the content of the PEM file
$pemContent = Get-Content $pemFilePath -Raw

# Convert the PEM content to a Base64 string
$base64Content = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($pemContent))

# Output the Base64 string to the console
Write-Output $base64Content

# Optionally, you can save this Base64 string to a file
$base64FilePath = ".\certificate_base64.txt"
$base64Content | Out-File -FilePath $base64FilePath
