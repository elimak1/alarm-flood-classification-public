Clean notbook cells with powershell using
Get-ChildItem \*.ipynb | ForEach-Object { jupyter nbconvert --to notebook --inplace --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True $\_.Name }
