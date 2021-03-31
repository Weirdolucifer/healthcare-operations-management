import pandas as pd

def run(my_sheet):
    file_name = 'data/WaitData.Published.xlsx'
    df = pd.read_excel(file_name, sheet_name = my_sheet)

    df = df.dropna(axis=1, how='all')
    df = df.dropna()

    remove = df.columns[df.columns.str.startswith('x_')]
    df.drop(remove, axis=1, inplace=True)

    df = df.astype(float)
    df.to_pickle('data/'+my_sheet+'.pkl')
    print('{} Pickle File created'.format(my_sheet))

if __name__=='__main__':
    facility = ['F1', 'F2', 'F3', 'F4'] 
    for my_sheet in facility:
        run(my_sheet)
