from utilpackage.path import path_definition

BasePath, pathDataset, pathRawData, pathJoinedData, pathSavedModels = path_definition()

print(f'BasePath = {BasePath}')
print(f'pathDataset = {pathDataset}')
print(f'pathRawData = {pathRawData}')
print(f'pathJoinedData = {pathJoinedData}')
print(f'pathSavedModels = {pathSavedModels}')