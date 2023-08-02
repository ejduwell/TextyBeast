import wsprPyannoteTest1
import sys

#inFile="/mnt/76cfc2bb-b830-47a1-9a9c-0d28b8a1efab/python_projects/installTesting/latest/dataScraping/input/JimmyKimmelAsksPresidentBarackObamaAboutHisDailyLife-QmPLGt5rd_k.mp4"
#OutDir="/mnt/76cfc2bb-b830-47a1-9a9c-0d28b8a1efab/python_projects/installTesting/latest/dataScraping/output"
#OutDir_sub="test43"


inFile = sys.argv[1];
OutDir = sys.argv[2];
OutDir_sub = sys.argv[3];

wsprPyannoteTest1.run(inFile,OutDir,OutDir_sub)
