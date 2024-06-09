## Directors Database

The directors database is processed from a raw database of 10-K filings. <br>
<br>
The 10-K filings are parsed to extract the directors' names and the companies they are affiliated with.
<br>
The database can be found at the following link: https://zenodo.org/records/5589195#.YX5a155BxPY
<br>
<br>
It's important to note that this database is about 30GB in size. Hence, we won't upload it to this repository 
(but we encourage you to do so locally).
<br>
<br>
The database is build the following way:
<br>
-> Year
<br>
-> -> Year
<br>
-> -> -> Companies 10-K filing
<br>
<Br>
The fillings are in ```json``` format and easy to parse.
<Br>
<Br>
For further information, please refer to the code in the ```analyze_data``` directory.