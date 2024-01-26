# Pipelines and Pandas

This is a short post about how to use Scikit-Learn Pipelines so that you have 'Pandas in, pandas out'. I'll build a small data pipeline on the Ames Iowa housing dataset. The first thing is to import the dataset, and inspect it! The data has 82 columns which include 23 nominal, 23 ordinal, 14 discrete, and 20 continuous variables (and 2 additional observation identifiers).


```python
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
```


```python
ames = pd.read_csv('ames.csv')
```


```python
ames.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS.SubClass</th>
      <th>MS.Zoning</th>
      <th>Lot.Frontage</th>
      <th>Lot.Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot.Shape</th>
      <th>Land.Contour</th>
      <th>...</th>
      <th>Pool.Area</th>
      <th>Pool.QC</th>
      <th>Fence</th>
      <th>Misc.Feature</th>
      <th>Misc.Val</th>
      <th>Mo.Sold</th>
      <th>Yr.Sold</th>
      <th>Sale.Type</th>
      <th>Sale.Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>526350040</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>526351010</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>172000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>526353030</td>
      <td>20</td>
      <td>RL</td>
      <td>93.0</td>
      <td>11160</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>244000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>527105010</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>189900</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>527105030</td>
      <td>60</td>
      <td>RL</td>
      <td>78.0</td>
      <td>9978</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>195500</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>527127150</td>
      <td>120</td>
      <td>RL</td>
      <td>41.0</td>
      <td>4920</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>213500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>527145080</td>
      <td>120</td>
      <td>RL</td>
      <td>43.0</td>
      <td>5005</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>191500</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>527146030</td>
      <td>120</td>
      <td>RL</td>
      <td>39.0</td>
      <td>5389</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>236500</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>527162130</td>
      <td>60</td>
      <td>RL</td>
      <td>60.0</td>
      <td>7500</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>189000</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 82 columns</p>
</div>



It's always a good idea to set an index on a DataFrame if you have one. In this case, the `PID` column is a unique identifier.


```python
ames = ames.set_index('PID').copy()
```


```python
ames.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Order</th>
      <td>2930.0</td>
      <td>1465.500000</td>
      <td>845.962470</td>
      <td>1.0</td>
      <td>733.25</td>
      <td>1465.5</td>
      <td>2197.75</td>
      <td>2930.0</td>
    </tr>
    <tr>
      <th>MS.SubClass</th>
      <td>2930.0</td>
      <td>57.387372</td>
      <td>42.638025</td>
      <td>20.0</td>
      <td>20.00</td>
      <td>50.0</td>
      <td>70.00</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>Lot.Frontage</th>
      <td>2440.0</td>
      <td>69.224590</td>
      <td>23.365335</td>
      <td>21.0</td>
      <td>58.00</td>
      <td>68.0</td>
      <td>80.00</td>
      <td>313.0</td>
    </tr>
    <tr>
      <th>Lot.Area</th>
      <td>2930.0</td>
      <td>10147.921843</td>
      <td>7880.017759</td>
      <td>1300.0</td>
      <td>7440.25</td>
      <td>9436.5</td>
      <td>11555.25</td>
      <td>215245.0</td>
    </tr>
    <tr>
      <th>Overall.Qual</th>
      <td>2930.0</td>
      <td>6.094881</td>
      <td>1.411026</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>Overall.Cond</th>
      <td>2930.0</td>
      <td>5.563140</td>
      <td>1.111537</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>5.0</td>
      <td>6.00</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>Year.Built</th>
      <td>2930.0</td>
      <td>1971.356314</td>
      <td>30.245361</td>
      <td>1872.0</td>
      <td>1954.00</td>
      <td>1973.0</td>
      <td>2001.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>Year.Remod.Add</th>
      <td>2930.0</td>
      <td>1984.266553</td>
      <td>20.860286</td>
      <td>1950.0</td>
      <td>1965.00</td>
      <td>1993.0</td>
      <td>2004.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>Mas.Vnr.Area</th>
      <td>2907.0</td>
      <td>101.896801</td>
      <td>179.112611</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>164.00</td>
      <td>1600.0</td>
    </tr>
    <tr>
      <th>BsmtFin.SF.1</th>
      <td>2929.0</td>
      <td>442.629566</td>
      <td>455.590839</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>370.0</td>
      <td>734.00</td>
      <td>5644.0</td>
    </tr>
    <tr>
      <th>BsmtFin.SF.2</th>
      <td>2929.0</td>
      <td>49.722431</td>
      <td>169.168476</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1526.0</td>
    </tr>
    <tr>
      <th>Bsmt.Unf.SF</th>
      <td>2929.0</td>
      <td>559.262547</td>
      <td>439.494153</td>
      <td>0.0</td>
      <td>219.00</td>
      <td>466.0</td>
      <td>802.00</td>
      <td>2336.0</td>
    </tr>
    <tr>
      <th>Total.Bsmt.SF</th>
      <td>2929.0</td>
      <td>1051.614544</td>
      <td>440.615067</td>
      <td>0.0</td>
      <td>793.00</td>
      <td>990.0</td>
      <td>1302.00</td>
      <td>6110.0</td>
    </tr>
    <tr>
      <th>X1st.Flr.SF</th>
      <td>2930.0</td>
      <td>1159.557679</td>
      <td>391.890885</td>
      <td>334.0</td>
      <td>876.25</td>
      <td>1084.0</td>
      <td>1384.00</td>
      <td>5095.0</td>
    </tr>
    <tr>
      <th>X2nd.Flr.SF</th>
      <td>2930.0</td>
      <td>335.455973</td>
      <td>428.395715</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>703.75</td>
      <td>2065.0</td>
    </tr>
    <tr>
      <th>Low.Qual.Fin.SF</th>
      <td>2930.0</td>
      <td>4.676792</td>
      <td>46.310510</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1064.0</td>
    </tr>
    <tr>
      <th>Gr.Liv.Area</th>
      <td>2930.0</td>
      <td>1499.690444</td>
      <td>505.508887</td>
      <td>334.0</td>
      <td>1126.00</td>
      <td>1442.0</td>
      <td>1742.75</td>
      <td>5642.0</td>
    </tr>
    <tr>
      <th>Bsmt.Full.Bath</th>
      <td>2928.0</td>
      <td>0.431352</td>
      <td>0.524820</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Bsmt.Half.Bath</th>
      <td>2928.0</td>
      <td>0.061134</td>
      <td>0.245254</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Full.Bath</th>
      <td>2930.0</td>
      <td>1.566553</td>
      <td>0.552941</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Half.Bath</th>
      <td>2930.0</td>
      <td>0.379522</td>
      <td>0.502629</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Bedroom.AbvGr</th>
      <td>2930.0</td>
      <td>2.854266</td>
      <td>0.827731</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Kitchen.AbvGr</th>
      <td>2930.0</td>
      <td>1.044369</td>
      <td>0.214076</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>TotRms.AbvGrd</th>
      <td>2930.0</td>
      <td>6.443003</td>
      <td>1.572964</td>
      <td>2.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>Fireplaces</th>
      <td>2930.0</td>
      <td>0.599317</td>
      <td>0.647921</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Garage.Yr.Blt</th>
      <td>2771.0</td>
      <td>1978.132443</td>
      <td>25.528411</td>
      <td>1895.0</td>
      <td>1960.00</td>
      <td>1979.0</td>
      <td>2002.00</td>
      <td>2207.0</td>
    </tr>
    <tr>
      <th>Garage.Cars</th>
      <td>2929.0</td>
      <td>1.766815</td>
      <td>0.760566</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Garage.Area</th>
      <td>2929.0</td>
      <td>472.819734</td>
      <td>215.046549</td>
      <td>0.0</td>
      <td>320.00</td>
      <td>480.0</td>
      <td>576.00</td>
      <td>1488.0</td>
    </tr>
    <tr>
      <th>Wood.Deck.SF</th>
      <td>2930.0</td>
      <td>93.751877</td>
      <td>126.361562</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>168.00</td>
      <td>1424.0</td>
    </tr>
    <tr>
      <th>Open.Porch.SF</th>
      <td>2930.0</td>
      <td>47.533447</td>
      <td>67.483400</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>27.0</td>
      <td>70.00</td>
      <td>742.0</td>
    </tr>
    <tr>
      <th>Enclosed.Porch</th>
      <td>2930.0</td>
      <td>23.011604</td>
      <td>64.139059</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1012.0</td>
    </tr>
    <tr>
      <th>X3Ssn.Porch</th>
      <td>2930.0</td>
      <td>2.592491</td>
      <td>25.141331</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>508.0</td>
    </tr>
    <tr>
      <th>Screen.Porch</th>
      <td>2930.0</td>
      <td>16.002048</td>
      <td>56.087370</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>576.0</td>
    </tr>
    <tr>
      <th>Pool.Area</th>
      <td>2930.0</td>
      <td>2.243345</td>
      <td>35.597181</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>800.0</td>
    </tr>
    <tr>
      <th>Misc.Val</th>
      <td>2930.0</td>
      <td>50.635154</td>
      <td>566.344288</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>17000.0</td>
    </tr>
    <tr>
      <th>Mo.Sold</th>
      <td>2930.0</td>
      <td>6.216041</td>
      <td>2.714492</td>
      <td>1.0</td>
      <td>4.00</td>
      <td>6.0</td>
      <td>8.00</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>Yr.Sold</th>
      <td>2930.0</td>
      <td>2007.790444</td>
      <td>1.316613</td>
      <td>2006.0</td>
      <td>2007.00</td>
      <td>2008.0</td>
      <td>2009.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>SalePrice</th>
      <td>2930.0</td>
      <td>180796.060068</td>
      <td>79886.692357</td>
      <td>12789.0</td>
      <td>129500.00</td>
      <td>160000.0</td>
      <td>213500.00</td>
      <td>755000.0</td>
    </tr>
  </tbody>
</table>
</div>



The main comment is that there's a lof of missing data, and that some columns should be dropped entirely (in particular Alley, Pool QC, Fence, Misc Feature, Fireplace QC) - or the dataset documentation needs to be checked to see that N/A isn't a default category in these cases. There's also a mixture of categorical and numerical features, which is a little tricky to handle. Luckily sklearn has the `FeatureUnion` and `Pipeline` objects to help us.


```python
ames.isnull().sum()
```




    Order                0
    MS.SubClass          0
    MS.Zoning            0
    Lot.Frontage       490
    Lot.Area             0
    Street               0
    Alley             2732
    Lot.Shape            0
    Land.Contour         0
    Utilities            0
    Lot.Config           0
    Land.Slope           0
    Neighborhood         0
    Condition.1          0
    Condition.2          0
    Bldg.Type            0
    House.Style          0
    Overall.Qual         0
    Overall.Cond         0
    Year.Built           0
    Year.Remod.Add       0
    Roof.Style           0
    Roof.Matl            0
    Exterior.1st         0
    Exterior.2nd         0
    Mas.Vnr.Type        23
    Mas.Vnr.Area        23
    Exter.Qual           0
    Exter.Cond           0
    Foundation           0
                      ... 
    Bedroom.AbvGr        0
    Kitchen.AbvGr        0
    Kitchen.Qual         0
    TotRms.AbvGrd        0
    Functional           0
    Fireplaces           0
    Fireplace.Qu      1422
    Garage.Type        157
    Garage.Yr.Blt      159
    Garage.Finish      159
    Garage.Cars          1
    Garage.Area          1
    Garage.Qual        159
    Garage.Cond        159
    Paved.Drive          0
    Wood.Deck.SF         0
    Open.Porch.SF        0
    Enclosed.Porch       0
    X3Ssn.Porch          0
    Screen.Porch         0
    Pool.Area            0
    Pool.QC           2917
    Fence             2358
    Misc.Feature      2824
    Misc.Val             0
    Mo.Sold              0
    Yr.Sold              0
    Sale.Type            0
    Sale.Condition       0
    SalePrice            0
    Length: 81, dtype: int64



First of all, let's build a transformer which drops the columns we suggest. A sklearn compatible transformer is a class which has to have two methods `fit` (which returns `self`), and `transform` (which can return whatever you want). It's a good idea to inherit from `sklearn.base.TransformerMixin` and `sklearn.base.BaseEstimator`. The general pattern of a Transformer is:


```python
from sklearn.base import BaseEstimator, TransformerMixin

class ExampleTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return do_something_to(self, X)
```

These are actually quite simple, and can be quite flexible. Later on, we might see a transformer that uses the `fit` method. Anyway, here's a column dropper. 


```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LarsCV
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
```


```python
class ColumnDropper(BaseEstimator, TransformerMixin):
    '''
    Transformer to drop a list of cols
    '''
    
    def __init__(self, drop_cols):
        self._drop_cols = drop_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df = df.drop(self._drop_cols, axis=1)
        return df
```


```python
y = ames['SalePrice'].copy()
X = ames.drop('SalePrice', axis=1).copy()
X.columns
```




    Index(['Order', 'MS.SubClass', 'MS.Zoning', 'Lot.Frontage', 'Lot.Area',
           'Street', 'Alley', 'Lot.Shape', 'Land.Contour', 'Utilities',
           'Lot.Config', 'Land.Slope', 'Neighborhood', 'Condition.1',
           'Condition.2', 'Bldg.Type', 'House.Style', 'Overall.Qual',
           'Overall.Cond', 'Year.Built', 'Year.Remod.Add', 'Roof.Style',
           'Roof.Matl', 'Exterior.1st', 'Exterior.2nd', 'Mas.Vnr.Type',
           'Mas.Vnr.Area', 'Exter.Qual', 'Exter.Cond', 'Foundation', 'Bsmt.Qual',
           'Bsmt.Cond', 'Bsmt.Exposure', 'BsmtFin.Type.1', 'BsmtFin.SF.1',
           'BsmtFin.Type.2', 'BsmtFin.SF.2', 'Bsmt.Unf.SF', 'Total.Bsmt.SF',
           'Heating', 'Heating.QC', 'Central.Air', 'Electrical', 'X1st.Flr.SF',
           'X2nd.Flr.SF', 'Low.Qual.Fin.SF', 'Gr.Liv.Area', 'Bsmt.Full.Bath',
           'Bsmt.Half.Bath', 'Full.Bath', 'Half.Bath', 'Bedroom.AbvGr',
           'Kitchen.AbvGr', 'Kitchen.Qual', 'TotRms.AbvGrd', 'Functional',
           'Fireplaces', 'Fireplace.Qu', 'Garage.Type', 'Garage.Yr.Blt',
           'Garage.Finish', 'Garage.Cars', 'Garage.Area', 'Garage.Qual',
           'Garage.Cond', 'Paved.Drive', 'Wood.Deck.SF', 'Open.Porch.SF',
           'Enclosed.Porch', 'X3Ssn.Porch', 'Screen.Porch', 'Pool.Area', 'Pool.QC',
           'Fence', 'Misc.Feature', 'Misc.Val', 'Mo.Sold', 'Yr.Sold', 'Sale.Type',
           'Sale.Condition'],
          dtype='object')




```python
pipe = Pipeline([('dropper', ColumnDropper(['Alley', 'Pool.QC', 'Fence', 'Misc.Feature', 'Fireplace.Qu', 'Order']))])
X_trans = pipe.fit_transform(X)
X_trans.columns
```




    Index(['MS.SubClass', 'MS.Zoning', 'Lot.Frontage', 'Lot.Area', 'Street',
           'Lot.Shape', 'Land.Contour', 'Utilities', 'Lot.Config', 'Land.Slope',
           'Neighborhood', 'Condition.1', 'Condition.2', 'Bldg.Type',
           'House.Style', 'Overall.Qual', 'Overall.Cond', 'Year.Built',
           'Year.Remod.Add', 'Roof.Style', 'Roof.Matl', 'Exterior.1st',
           'Exterior.2nd', 'Mas.Vnr.Type', 'Mas.Vnr.Area', 'Exter.Qual',
           'Exter.Cond', 'Foundation', 'Bsmt.Qual', 'Bsmt.Cond', 'Bsmt.Exposure',
           'BsmtFin.Type.1', 'BsmtFin.SF.1', 'BsmtFin.Type.2', 'BsmtFin.SF.2',
           'Bsmt.Unf.SF', 'Total.Bsmt.SF', 'Heating', 'Heating.QC', 'Central.Air',
           'Electrical', 'X1st.Flr.SF', 'X2nd.Flr.SF', 'Low.Qual.Fin.SF',
           'Gr.Liv.Area', 'Bsmt.Full.Bath', 'Bsmt.Half.Bath', 'Full.Bath',
           'Half.Bath', 'Bedroom.AbvGr', 'Kitchen.AbvGr', 'Kitchen.Qual',
           'TotRms.AbvGrd', 'Functional', 'Fireplaces', 'Garage.Type',
           'Garage.Yr.Blt', 'Garage.Finish', 'Garage.Cars', 'Garage.Area',
           'Garage.Qual', 'Garage.Cond', 'Paved.Drive', 'Wood.Deck.SF',
           'Open.Porch.SF', 'Enclosed.Porch', 'X3Ssn.Porch', 'Screen.Porch',
           'Pool.Area', 'Misc.Val', 'Mo.Sold', 'Yr.Sold', 'Sale.Type',
           'Sale.Condition'],
          dtype='object')



Another option, especially with columns with missing values, is to impute the value but to include a column telling the model where the imputed values are.


```python
class ImputeWithDummy(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols_to_impute, strategy, fill='NA'):
        self.cols_to_impute = cols_to_impute
        self.strategy = strategy
        self.fill = fill
        
    def fit(self, X, y=None, **kwargs):
        if self.strategy == 'mean':
            self.fill = X.mean()
        elif self.strategy == 'median':
            self.fill = X.median()
        elif self.strategy == 'mode':
            self.fill = X.mode().iloc[0]
        elif self.strategy == 'fill':
            if type(self.fill) is list and type(X) is pd.DataFrame:
                self.fill = dict([(cname, v) for cname,v in zip(X.columns, self.fill)])
        return self
    
    def transform(self, X):
        df = X.copy()
        for col in self.cols_to_impute:
            df['{}_missing'.format(col)] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(self.fill[col])
        return df
    
X = pd.read_csv('ames.csv')
imputer = ImputeWithDummy(['Alley'], strategy='mode')
X_transformed = imputer.fit_transform(X)
X_transformed[['Alley', 'Alley_missing']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alley</th>
      <th>Alley_missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Grvl</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Grvl</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grvl</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grvl</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Grvl</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Of course, you should always read the data documentation (https://ww2.amstat.org/publications/jse/v19n3/decock/datadocumentation.txt), and there you'll see for Alley that NaN means No Alley Access, and that we don't need to any imputation at all!


```python
class NaNImpute(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols, fill_vals):
        self.cols = cols
        self.fill_vals = fill_vals
        
    def fit(self, X, y=None, **kwargs):
        return self
    
    def transform(self, X):
        df = X.copy()
        for i, col in enumerate(self.cols):
            df[col].fillna(self.fill_vals[i])
```

The other thing we'll need to consider is that some columns will need to be converted to numeric features first, before an estimator can be fitted. First we'll fit an imputer, and then encode.


```python
class DummyEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):

        self.columns = columns

    def fit(self, X, y=None, **kwargs):
        return self
    
    def transform(self, X, y=None, **kwargs):
        return pd.get_dummies(X, columns=self.columns, drop_first=True)
```


```python
impute_cols = ['Alley', 'Pool.QC', 'Fence', 'Misc.Feature', 'Fireplace.Qu']
pipe = Pipeline([('impute', ImputeWithDummy(impute_cols, strategy='mode')), ('encode', DummyEncoding(impute_cols))])
X = pd.read_csv('ames.csv')
X_trans = pipe.fit_transform(X)
X_trans.columns
```




    Index(['Order', 'PID', 'MS.SubClass', 'MS.Zoning', 'Lot.Frontage', 'Lot.Area',
           'Street', 'Lot.Shape', 'Land.Contour', 'Utilities', 'Lot.Config',
           'Land.Slope', 'Neighborhood', 'Condition.1', 'Condition.2', 'Bldg.Type',
           'House.Style', 'Overall.Qual', 'Overall.Cond', 'Year.Built',
           'Year.Remod.Add', 'Roof.Style', 'Roof.Matl', 'Exterior.1st',
           'Exterior.2nd', 'Mas.Vnr.Type', 'Mas.Vnr.Area', 'Exter.Qual',
           'Exter.Cond', 'Foundation', 'Bsmt.Qual', 'Bsmt.Cond', 'Bsmt.Exposure',
           'BsmtFin.Type.1', 'BsmtFin.SF.1', 'BsmtFin.Type.2', 'BsmtFin.SF.2',
           'Bsmt.Unf.SF', 'Total.Bsmt.SF', 'Heating', 'Heating.QC', 'Central.Air',
           'Electrical', 'X1st.Flr.SF', 'X2nd.Flr.SF', 'Low.Qual.Fin.SF',
           'Gr.Liv.Area', 'Bsmt.Full.Bath', 'Bsmt.Half.Bath', 'Full.Bath',
           'Half.Bath', 'Bedroom.AbvGr', 'Kitchen.AbvGr', 'Kitchen.Qual',
           'TotRms.AbvGrd', 'Functional', 'Fireplaces', 'Garage.Type',
           'Garage.Yr.Blt', 'Garage.Finish', 'Garage.Cars', 'Garage.Area',
           'Garage.Qual', 'Garage.Cond', 'Paved.Drive', 'Wood.Deck.SF',
           'Open.Porch.SF', 'Enclosed.Porch', 'X3Ssn.Porch', 'Screen.Porch',
           'Pool.Area', 'Misc.Val', 'Mo.Sold', 'Yr.Sold', 'Sale.Type',
           'Sale.Condition', 'SalePrice', 'Alley_missing', 'Pool.QC_missing',
           'Fence_missing', 'Misc.Feature_missing', 'Fireplace.Qu_missing',
           'Alley_Pave', 'Pool.QC_Fa', 'Pool.QC_Gd', 'Pool.QC_TA', 'Fence_GdWo',
           'Fence_MnPrv', 'Fence_MnWw', 'Misc.Feature_Gar2', 'Misc.Feature_Othr',
           'Misc.Feature_Shed', 'Misc.Feature_TenC', 'Fireplace.Qu_Fa',
           'Fireplace.Qu_Gd', 'Fireplace.Qu_Po', 'Fireplace.Qu_TA'],
          dtype='object')



This dataset has several types of columns - continuous features encoded as ints and floats, but also some ordinal variables have snick in as ints. We'll properly define all the int and float colukmns first:


```python
float_cols = ['Lot.Frontage', 
              'Mas.Vnr.Area', 
              'BsmtFin.SF.1', 
              'BsmtFin.SF.2', 
              'Bsmt.Unf.SF',
              'Total.Bsmt.SF',
              'Garage.Cars',
              'Garage.Area'
             ]

int_cols = ['MS.SubClass', 
            'Lot.Area',
            'X1st.Flr.SF',
            'X2nd.Flr.SF', 
            'Low.Qual.Fin.SF', 
            'Gr.Liv.Area', 
            'Full.Bath',
            'Half.Bath', 
            'Bedroom.AbvGr', 
            'Kitchen.AbvGr', 
            'TotRms.AbvGrd',
            'Fireplaces', 
            'Wood.Deck.SF', 
            'Open.Porch.SF', 
            'Enclosed.Porch',
            'X3Ssn.Porch', 
            'Screen.Porch', 
            'Pool.Area', 
            'Misc.Val'
           ]
```

Finally, we need some way to deal with ordinal features:

  - Lot Shape (Ordinal): General shape of property
  - Utilities (Ordinal): Type of utilities available
  - Land Slope (Ordinal): Slope of property
  - Overall Qual (Ordinal): Rates the overall material and finish of the house
  - Overall Cond (Ordinal): Rates the overall condition of the house
  - Exter Qual (Ordinal): Evaluates the quality of the material on the exterior
  - Exter Cond (Ordinal): Evaluates the present condition of the material on the exterior
  - Bsmt Qual (Ordinal): Evaluates the height of the basement
  - Bsmt Cond (Ordinal): Evaluates the general condition of the basement
  - Bsmt Exposure	(Ordinal): Refers to walkout or garden level walls
  - BsmtFin Type 1	(Ordinal): Rating of basement finished area
  - BsmtFin Type 2	(Ordinal): Rating of basement finished area (if multiple types)
  - HeatingQC (Ordinal): Heating quality and condition
  - Electrical (Ordinal): Electrical system
  - FireplaceQu (Ordinal): Fireplace quality
  - Garage Finish (Ordinal)	: Interior finish of the garage
  - Garage Qual (Ordinal): Garage quality
  - Garage Cond (Ordinal): Garage condition
  - Paved Drive (Ordinal): Paved driveway
  - Pool QC (Ordinal): Pool quality
  - Fence (Ordinal): Fence quality

To do this we could use the OrdinalEncoder from http://contrib.scikit-learn.org/categorical-encoding/, which will be included in sklearn in a future release  - but I have trouble getting this to work with Pandas. Another choice is just to write our own. What we'll do instead is to mix ordinal and categorical variables, and use the OneHotEncoder from the category_encoders package.


```python
ord_cols     = ['Lot.Shape',
                'Utilities',
                'Land.Slope',
                'Overall.Qual',
                'Overall.Cond',
                'Exter.Qual', 
                'Exter.Cond',
                'Bsmt.Qual',
                'Bsmt.Cond', 
                'Bsmt.Exposure', 
                'BsmtFin.Type.1', 
                'BsmtFin.SF.1',
                'Heating.QC',
                'Electrical',
                'Fireplace.Qu',
                'Garage.Finish',
                'Garage.Qual',
                'Garage.Cond',
                'Paved.Drive',
                'Pool.QC',
                'Fence',
               ]

cat_cols = ['MS.SubClass',
            'MS.Zoning',
            'Street',
            'Alley',
            'Land.Contour',
            'Lot.Config',
            'Neighborhood',
            'Condition.1',
            'Condition.2',
            'Bldg.Type',
            'House.Style',
            'Roof.Style',
            'Exterior.1st', 
            'Exterior.2nd',
            'Mas.Vnr.Type',
            'Foundation',
            'Heating',
            'Central.Air',
            'Garage.Type',
            'Misc.Feature',
            'Sale.Type',
            'Sale.Condition'
]
```

Finally, we define a few useful transforms and put together our first pipeline.


```python
class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''
    Select columns from pandas dataframe by specifying a list of column names
    '''
    def __init__(self, col_names):
        self.col_names = col_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.col_names]
```


```python
class Scale(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols):
        self.scaler = StandardScaler()
        self.cols = cols
        self.index = []
        
    def fit(self, X, y=None, **kwargs):
        self.scaler.fit(X)
        self.cols = X.columns
        self.index = X.index
        return self
        
    def transform(self, X):
        df = X.copy()
        df = self.scaler.transform(df)
        df = pd.DataFrame(df, columns=self.cols, index=self.index)
        return df
```


```python
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from scipy import sparse


class FeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, weight, X, y,
                                        **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, X)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs
```


```python
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from category_encoders import OneHotEncoder

numerical_cols = int_cols + float_cols

pipe = Pipeline([
    ('features', FeatureUnion(n_jobs=1, transformer_list=[
        ('numericals', Pipeline([
             ('selector', DataFrameSelector(numerical_cols)),
             ('imputer', ImputeWithDummy(numerical_cols, strategy='mean')),
             #('scaling', Scale(numerical_cols))
        ])),
        
        #('categoricals', Pipeline([
        #     ('selector', DataFrameSelector(cat_cols)), 
        #     ('encode', OneHotEncoder(cat_cols, return_df=True)),
        #])),
        
        #('NanImpute', Pipeline([
        ##     ('selector', DataFrameSelector(['Alley', 'Pool.QC', 'Fence', 'Misc.Feature', 'Fireplace.Qu'])),
        #     ('nan_impute', NaNImpute(['Alley', 'Pool.QC', 'Fence', 'Misc.Feature', 'Fireplace.Qu'], 'Not Applicable'))
        #])),
    ])), 
])  
```


```python
X = pd.read_csv('ames.csv')
X = X.set_index('PID').copy()
```


```python
X_trans = pipe.fit_transform(X)
X_trans
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MS.SubClass</th>
      <th>Lot.Area</th>
      <th>X1st.Flr.SF</th>
      <th>X2nd.Flr.SF</th>
      <th>Low.Qual.Fin.SF</th>
      <th>Gr.Liv.Area</th>
      <th>Full.Bath</th>
      <th>Half.Bath</th>
      <th>Bedroom.AbvGr</th>
      <th>Kitchen.AbvGr</th>
      <th>...</th>
      <th>Pool.Area_missing</th>
      <th>Misc.Val_missing</th>
      <th>Lot.Frontage_missing</th>
      <th>Mas.Vnr.Area_missing</th>
      <th>BsmtFin.SF.1_missing</th>
      <th>BsmtFin.SF.2_missing</th>
      <th>Bsmt.Unf.SF_missing</th>
      <th>Total.Bsmt.SF_missing</th>
      <th>Garage.Cars_missing</th>
      <th>Garage.Area_missing</th>
    </tr>
    <tr>
      <th>PID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>526301100</th>
      <td>20</td>
      <td>31770</td>
      <td>1656</td>
      <td>0</td>
      <td>0</td>
      <td>1656</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>526350040</th>
      <td>20</td>
      <td>11622</td>
      <td>896</td>
      <td>0</td>
      <td>0</td>
      <td>896</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>526351010</th>
      <td>20</td>
      <td>14267</td>
      <td>1329</td>
      <td>0</td>
      <td>0</td>
      <td>1329</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>526353030</th>
      <td>20</td>
      <td>11160</td>
      <td>2110</td>
      <td>0</td>
      <td>0</td>
      <td>2110</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527105010</th>
      <td>60</td>
      <td>13830</td>
      <td>928</td>
      <td>701</td>
      <td>0</td>
      <td>1629</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527105030</th>
      <td>60</td>
      <td>9978</td>
      <td>926</td>
      <td>678</td>
      <td>0</td>
      <td>1604</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527127150</th>
      <td>120</td>
      <td>4920</td>
      <td>1338</td>
      <td>0</td>
      <td>0</td>
      <td>1338</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527145080</th>
      <td>120</td>
      <td>5005</td>
      <td>1280</td>
      <td>0</td>
      <td>0</td>
      <td>1280</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527146030</th>
      <td>120</td>
      <td>5389</td>
      <td>1616</td>
      <td>0</td>
      <td>0</td>
      <td>1616</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527162130</th>
      <td>60</td>
      <td>7500</td>
      <td>1028</td>
      <td>776</td>
      <td>0</td>
      <td>1804</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527163010</th>
      <td>60</td>
      <td>10000</td>
      <td>763</td>
      <td>892</td>
      <td>0</td>
      <td>1655</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527165230</th>
      <td>20</td>
      <td>7980</td>
      <td>1187</td>
      <td>0</td>
      <td>0</td>
      <td>1187</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527166040</th>
      <td>60</td>
      <td>8402</td>
      <td>789</td>
      <td>676</td>
      <td>0</td>
      <td>1465</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527180040</th>
      <td>20</td>
      <td>10176</td>
      <td>1341</td>
      <td>0</td>
      <td>0</td>
      <td>1341</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527182190</th>
      <td>120</td>
      <td>6820</td>
      <td>1502</td>
      <td>0</td>
      <td>0</td>
      <td>1502</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527216070</th>
      <td>60</td>
      <td>53504</td>
      <td>1690</td>
      <td>1589</td>
      <td>0</td>
      <td>3279</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527225035</th>
      <td>50</td>
      <td>12134</td>
      <td>1080</td>
      <td>672</td>
      <td>0</td>
      <td>1752</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527258010</th>
      <td>20</td>
      <td>11394</td>
      <td>1856</td>
      <td>0</td>
      <td>0</td>
      <td>1856</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527276150</th>
      <td>20</td>
      <td>19138</td>
      <td>864</td>
      <td>0</td>
      <td>0</td>
      <td>864</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527302110</th>
      <td>20</td>
      <td>13175</td>
      <td>2073</td>
      <td>0</td>
      <td>0</td>
      <td>2073</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527358140</th>
      <td>20</td>
      <td>11751</td>
      <td>1844</td>
      <td>0</td>
      <td>0</td>
      <td>1844</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527358200</th>
      <td>85</td>
      <td>10625</td>
      <td>1173</td>
      <td>0</td>
      <td>0</td>
      <td>1173</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527368020</th>
      <td>60</td>
      <td>7500</td>
      <td>814</td>
      <td>860</td>
      <td>0</td>
      <td>1674</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527402200</th>
      <td>20</td>
      <td>11241</td>
      <td>1004</td>
      <td>0</td>
      <td>0</td>
      <td>1004</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527402250</th>
      <td>20</td>
      <td>12537</td>
      <td>1078</td>
      <td>0</td>
      <td>0</td>
      <td>1078</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527403020</th>
      <td>20</td>
      <td>8450</td>
      <td>1056</td>
      <td>0</td>
      <td>0</td>
      <td>1056</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527404120</th>
      <td>20</td>
      <td>8400</td>
      <td>882</td>
      <td>0</td>
      <td>0</td>
      <td>882</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527425090</th>
      <td>20</td>
      <td>10500</td>
      <td>864</td>
      <td>0</td>
      <td>0</td>
      <td>864</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527427230</th>
      <td>120</td>
      <td>5858</td>
      <td>1337</td>
      <td>0</td>
      <td>0</td>
      <td>1337</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>527451180</th>
      <td>160</td>
      <td>1680</td>
      <td>483</td>
      <td>504</td>
      <td>0</td>
      <td>987</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>916477010</th>
      <td>20</td>
      <td>13618</td>
      <td>1960</td>
      <td>0</td>
      <td>0</td>
      <td>1960</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>921205030</th>
      <td>20</td>
      <td>11443</td>
      <td>2028</td>
      <td>0</td>
      <td>0</td>
      <td>2028</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>921205050</th>
      <td>20</td>
      <td>11577</td>
      <td>1838</td>
      <td>0</td>
      <td>0</td>
      <td>1838</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923125030</th>
      <td>20</td>
      <td>31250</td>
      <td>1600</td>
      <td>0</td>
      <td>0</td>
      <td>1600</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923202025</th>
      <td>90</td>
      <td>7020</td>
      <td>1368</td>
      <td>0</td>
      <td>0</td>
      <td>1368</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923203090</th>
      <td>120</td>
      <td>4500</td>
      <td>1216</td>
      <td>0</td>
      <td>0</td>
      <td>1216</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923203100</th>
      <td>120</td>
      <td>4500</td>
      <td>1337</td>
      <td>0</td>
      <td>0</td>
      <td>1337</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923205120</th>
      <td>20</td>
      <td>17217</td>
      <td>1140</td>
      <td>0</td>
      <td>0</td>
      <td>1140</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923225190</th>
      <td>160</td>
      <td>2665</td>
      <td>616</td>
      <td>688</td>
      <td>0</td>
      <td>1304</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923225240</th>
      <td>160</td>
      <td>2665</td>
      <td>925</td>
      <td>550</td>
      <td>0</td>
      <td>1475</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923225260</th>
      <td>160</td>
      <td>3964</td>
      <td>1291</td>
      <td>1230</td>
      <td>0</td>
      <td>2521</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923225510</th>
      <td>20</td>
      <td>10172</td>
      <td>874</td>
      <td>0</td>
      <td>0</td>
      <td>874</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923226150</th>
      <td>90</td>
      <td>11836</td>
      <td>1652</td>
      <td>0</td>
      <td>0</td>
      <td>1652</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923226180</th>
      <td>180</td>
      <td>1470</td>
      <td>630</td>
      <td>0</td>
      <td>0</td>
      <td>630</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923226290</th>
      <td>160</td>
      <td>1484</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923227100</th>
      <td>20</td>
      <td>13384</td>
      <td>1360</td>
      <td>0</td>
      <td>0</td>
      <td>1360</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923228130</th>
      <td>180</td>
      <td>1533</td>
      <td>630</td>
      <td>0</td>
      <td>0</td>
      <td>630</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923228180</th>
      <td>160</td>
      <td>1533</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923228210</th>
      <td>160</td>
      <td>1526</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923228260</th>
      <td>160</td>
      <td>1936</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923228310</th>
      <td>160</td>
      <td>1894</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923229110</th>
      <td>90</td>
      <td>12640</td>
      <td>1728</td>
      <td>0</td>
      <td>0</td>
      <td>1728</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923230040</th>
      <td>90</td>
      <td>9297</td>
      <td>1728</td>
      <td>0</td>
      <td>0</td>
      <td>1728</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923250060</th>
      <td>20</td>
      <td>17400</td>
      <td>1126</td>
      <td>0</td>
      <td>0</td>
      <td>1126</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923251180</th>
      <td>20</td>
      <td>20000</td>
      <td>1224</td>
      <td>0</td>
      <td>0</td>
      <td>1224</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923275080</th>
      <td>80</td>
      <td>7937</td>
      <td>1003</td>
      <td>0</td>
      <td>0</td>
      <td>1003</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923276100</th>
      <td>20</td>
      <td>8885</td>
      <td>902</td>
      <td>0</td>
      <td>0</td>
      <td>902</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>923400125</th>
      <td>85</td>
      <td>10441</td>
      <td>970</td>
      <td>0</td>
      <td>0</td>
      <td>970</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>924100070</th>
      <td>20</td>
      <td>10010</td>
      <td>1389</td>
      <td>0</td>
      <td>0</td>
      <td>1389</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>924151050</th>
      <td>60</td>
      <td>9627</td>
      <td>996</td>
      <td>1004</td>
      <td>0</td>
      <td>2000</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2930 rows × 54 columns</p>
</div>




```python
l = list(X_trans.isnull().sum())
```


```python
import warnings
warnings.filterwarnings("ignore")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
model = Pipeline([('pipeline', pipe), ('clf', LarsCV())])
model.fit(X_train, y_train)
```




    Pipeline(memory=None,
         steps=[('pipeline', Pipeline(memory=None,
         steps=[('features', FeatureUnion(n_jobs=1,
           transformer_list=[('numericals', Pipeline(memory=None,
         steps=[('selector', DataFrameSelector(col_names=['MS.SubClass', 'Lot.Area', 'X1st.Flr.SF', 'X2nd.Flr.SF', 'Low.Qual.Fin.SF', 'Gr.Liv.Area', 'Fu...max_n_alphas=1000, n_jobs=1, normalize=True,
        positive=False, precompute='auto', verbose=False))])




```python
pipe.fit_transform(X_test).columns
```




    Index(['MS.SubClass', 'Lot.Area', 'X1st.Flr.SF', 'X2nd.Flr.SF',
           'Low.Qual.Fin.SF', 'Gr.Liv.Area', 'Full.Bath', 'Half.Bath',
           'Bedroom.AbvGr', 'Kitchen.AbvGr', 'TotRms.AbvGrd', 'Fireplaces',
           'Wood.Deck.SF', 'Open.Porch.SF', 'Enclosed.Porch', 'X3Ssn.Porch',
           'Screen.Porch', 'Pool.Area', 'Misc.Val', 'Lot.Frontage', 'Mas.Vnr.Area',
           'BsmtFin.SF.1', 'BsmtFin.SF.2', 'Bsmt.Unf.SF', 'Total.Bsmt.SF',
           'Garage.Cars', 'Garage.Area', 'MS.SubClass_missing', 'Lot.Area_missing',
           'X1st.Flr.SF_missing', 'X2nd.Flr.SF_missing', 'Low.Qual.Fin.SF_missing',
           'Gr.Liv.Area_missing', 'Full.Bath_missing', 'Half.Bath_missing',
           'Bedroom.AbvGr_missing', 'Kitchen.AbvGr_missing',
           'TotRms.AbvGrd_missing', 'Fireplaces_missing', 'Wood.Deck.SF_missing',
           'Open.Porch.SF_missing', 'Enclosed.Porch_missing',
           'X3Ssn.Porch_missing', 'Screen.Porch_missing', 'Pool.Area_missing',
           'Misc.Val_missing', 'Lot.Frontage_missing', 'Mas.Vnr.Area_missing',
           'BsmtFin.SF.1_missing', 'BsmtFin.SF.2_missing', 'Bsmt.Unf.SF_missing',
           'Total.Bsmt.SF_missing', 'Garage.Cars_missing', 'Garage.Area_missing'],
          dtype='object')




```python
preds = model.predict(X_test)
```


```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def rmse_cv(model, X, y):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=2))
    return(rmse)
```


```python
rmse_cv(model, X, y)
```




    array([37647.53964453, 71123.27879506])




```python
from sklearn.metrics import r2_score

def get_score(prediction, labels):    
    print('R2: {}'.format(r2_score(prediction, labels)))
```


```python
get_score(preds, y_test)
```

    R2: 0.6208433062823728



```python

```
