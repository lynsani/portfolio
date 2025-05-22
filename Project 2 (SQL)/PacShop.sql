--Question 1 --
-- Find the first and last order made by each buyer in each shop--

Select u.shopid, u.buyerid, min(o.order_time) as first_order, max(o.order_time) as last_order
from user_tab u
join order_tab o
	using(shopid)
group by 1, 2
order by 2 asc;

-- Question 2 --
-- Find buyers who placed orders more than once in a month--
with count_order as(
SELECT 
    buyerid, 
  	extract(month from order_time) AS order_month,
   	COUNT(orderid) AS total_order
FROM order_tab
join user_tab
	using(buyerid)
GROUP BY 1,2
HAVING COUNT(orderid) > 1
ORDER BY 1,2 ASC
)

select
	buyerid,
	avg(total_order)::int as avg_order
from count_order
group by 1


-- question 3 --
-- Find the first buyer in each shop --

select distinct (shopid)
	shopid,
	first_value(buyerid) over(partition by shopid order by order_time asc) as buyer_id,
	first_value(order_time) over(partition by shopid order by order_time asc) as order_time
from performance_tab
join order_tab
	using(shopid)


-- question 4 --
-- find top 10 buyer based on GMV with country code ID and SG--

select u.buyerid, u.country, sum(o.gmv)
from user_tab u
join order_tab o
	using(buyerid)
where u.country = 'ID' or u.country = 'SG'
group by 1,2
order by 3 desc
limit 10;

-- question 5 --
-- Find the number of buyers in each country who placed orders with odd and even item IDs-- 

SELECT 
    u.country,
    COUNT(CASE WHEN o.itemid % 2 = 0 THEN 1 END) AS even_count,
    COUNT(CASE WHEN o.itemid % 2 != 0 THEN 1 END) AS odd_count
FROM 
    user_tab u
JOIN 
    order_tab o
USING (buyerid)
GROUP BY 
    u.country;

-- question 6 --
-- analyze convertion rate & CTR in each shop

SELECT DISTINCT shopid, 
	n_order, 
	SUM(total_clicks) OVER(PARTITION BY shopid) sum_tc,  
	SUM(item_views) OVER(PARTITION BY shopid) sum_tv,
	SUM(impressions) OVER(PARTITION BY shopid) sum_imp,
		n_order::NUMERIC / SUM(item_views) OVER(PARTITION BY shopid)  AS cvr,
		SUM(total_clicks) OVER(PARTITION BY shopid) ::numeric/ SUM(impressions) 
		OVER(PARTITION BY shopid) AS ctr
FROM performance_tab pt 
JOIN (SELECT shopid, 
			COUNT(orderid) n_order 
			FROM order_tab ot GROUP BY shopid) using(shopid)
ORDER BY 1 asc;





