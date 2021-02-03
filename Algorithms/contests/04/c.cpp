#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

struct DSet
{
    int *p, *r;
    ll *exp;

    DSet(int n)
    {
        this -> p = new int[n];
        this -> r = new int[n];
        this -> exp = new ll[n];

        for (int i = 0; i < n; i++)
        {
            r[i] = 0;
            exp[i] = 0;
            p[i] = i;
        }
    }

    int get(int u)
    {
        if (u != p[u])
            // p[u] = get(p[u]);
            return get(p[u]);
        return p[u];
    }

    void add_exp(int x, int v)
    {
        exp[get(x)] += v;
    }

    ll get_exp(int x)
    {
        ll exp_x = exp[x];
        // cout << x << " " << p[x] << "\n";
        while(x != p[x])
        {
            x = p[x];
            exp_x += exp[x];
        }
        return exp_x;
    }

    void link(int x, int y)
    {
        int c1 = get(x);
        int c2 = get(y);

        if (c1 != c2)
        {
            if (r[c1] > r[c2])
            {
                p[c2] = c1;
                exp[c2] -= exp[c1];
            }
            else
            {
                p[c1] = c2;
                exp[c1] -= exp[c2];
            }

            if (r[c1] == r[c2])
                r[c2]++;
        }
    }
};





int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    int n, m;
    cin >> n >> m;

    DSet clans(n);


    vector<int> exp_out;
    for (int i = 0; i < m; i++)
    {
        string todo;
        int x;
        cin >> todo;
        if (todo == "join")
        {
            int y;
            cin >> x >> y;
            clans.link(x - 1, y - 1);
        }
        else if (todo == "add")
        {
            int v;
            cin >> x >> v;
            clans.add_exp(x - 1, v);
        }
        else
        {
            cin >> x;
            cout << clans.get_exp(x - 1) << "\n";
            // exp_out.push_back(clans.get_exp(x - 1));
        }
    }
    
    // for (auto i = exp_out.begin(); i != exp_out.end(); i++)
    // {
    //     cout << *i << "\n";
    // }
    
    
    return 0;
}