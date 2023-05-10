def get_fee(data):
    import numpy as np
    import pandas as pd
    
    out = data.copy()
    
    out["fee"] = np.where((data["PSP"] == "Moneycard") & (data["success"] == 1), 5, 
                    np.where((data["PSP"] == "Moneycard") & (data["success"] == 0), 2,
                    np.where((data["PSP"] == "Goldcard") & (data["success"] == 1), 10,
                    np.where((data["PSP"] == "Goldcard") & (data["success"] == 0), 5,
                    np.where((data["PSP"] == "UK_Card") & (data["success"] == 1), 3,
                    np.where((data["PSP"] == "UK_Card") & (data["success"] == 0), 1,
                    np.where((data["PSP"] == "Simplecard") & (data["success"] == 1), 1, 0.5)))))))
    
    return out

def getNumUpper(timestamp, lower, upper, amount, country, testdata):
    selecteddata = testdata[
        (testdata['amount'] == amount) & 
        (testdata['country'] == country) &
        (testdata['tmsp'] > timestamp) &
        (testdata['tmsp'] <= upper)
    ]
    
    return len(selecteddata)

def getLowerFailedPSP(row, woUpper, inFrame):
    if (woUpper.index.get_loc(row.name) == 0):
        locPrev = inFrame.iloc[0, :].name
    else:
        ilocPrev = woUpper.index.get_loc(row.name) - 1
        locPrev = woUpper.iloc[ilocPrev, :].name
    
    locAct = row.name
    
    selection = inFrame.loc[locPrev:locAct]
    selection = selection[
        (selection.numUpper > 0) &
        (selection.amount == row.amount) &
        (selection.country == row.country)
    ]
    if len(selection) > 0:
        failedPSP = sorted(list(selection["PSP"].unique()))
        returnDummy = []
        for psp in sorted(inFrame["PSP"].unique()):
            if any(psp in s for s in failedPSP):
                returnDummy.append(int(1))
            else:
                returnDummy.append(int(0))
        return [1, failedPSP, returnDummy[0], returnDummy[1], returnDummy[2], returnDummy[3]]
    else:
        return [0, [], 0, 0, 0, 0]

def selectRows(data):
    import pandas as pd
    import numpy as np
    import time
    
    start_time = time.time()
    out = data.copy()
    out["lower"] = out["tmsp"] + pd.Timedelta(minutes=-1)
    out["upper"] = out["tmsp"] + pd.Timedelta(minutes=1)
    
    out['numUpper'] = out.apply(lambda row : getNumUpper(
        row['tmsp'],
        row['lower'],
        row['upper'],
        row['amount'],
        row['country'],
        testdata = out
    ), axis = 1)
    out = out.copy().sort_values(by=['tmsp'])
    woUpper = out[out['numUpper'] == 0]
    print("= Half time: " + str(time.time() - start_time) + " seconds")
    out = out.join(
        woUpper.apply(lambda row : getLowerFailedPSP(
            row, 
            woUpper.copy(), 
            out.copy()
        ), axis = 1, result_type = 'expand').rename(columns={
            0: "failPrevious",
            1: "failedPSP",
            2: "failed_Goldcard",
            3: "failed_Moneycard",
            4: "failed_Simplecard",
            5: "failed_UK_Card"
        })
    )[out["numUpper"] == 0]
    out = out.drop(columns=['failedPSP', 'lower', 'upper', 'numUpper'])
    
    print("= End Time: " + str(time.time() - start_time) + " seconds")
    
    return out