/* I connected to the Oracle database using LIBNAME */
libname edw oracle user="user" password="ddcpassword" path="path" schema="schema";

/* I gathered the claim number data from three different sources */
proc sql;
   create table claim_data as
   select 
     t1.claim_number, 
     t1.filing_date, 
     t2.notation_date,
     t3.other_column
   from 
     edw.table1 as t1
   left join 
     edw.table2 as t2
     on t1.claim_number = t2.claim_number
   left join 
     edw.table3 as t3
     on t1.claim_number = t3.claim_number
   where t1.filing_date >= intnx('week', today(), -1, 'same')  /* claims filed in the past week */
     and t2.notation_date is null                              /* no notation date within 2 days */
     or t2.notation_date > t1.filing_date + 2;
quit;

/* I added a calculated column for days since claim was filed */
data claim_data_final;
   set claim_data;
   days_since_filing = today() - filing_date;
run;

/* I used PROC EXPORT to generate a CSV file and then email the report */
proc export data=claim_data_final
   outfile='/path/to/claim_report.csv'
   dbms=csv
   replace;
run;

/* I sent the report via email */
filename mymail email
   to=("recipient1@geico.com" "recipient2@geico.com")
   subject="Weekly Claim Report"
   from="ray@geico.com";

data _null_;
   file mymail;
   put 'Please find attached the weekly claim report.';
run;
