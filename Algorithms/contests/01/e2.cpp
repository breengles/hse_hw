#include <bits/stdc++.h>

using namespace std;

#define ll long long
#define ul unsigned long
#define ull unsigned long long

bool fc(const pair<float, int> &x, const pair<float, int> &y){
    return x.first < y.first;
}

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    int n, k;
    cin >> n >> k;

    vector<pair<int, int>> a(n);

    float l = 0;
    float r = -1;
    for (int i = 0; i < n; i++){
        int v, w;
        cin >> v >> w;
        a[i] = make_pair(v, w);
        if (a[i].first/a[i].second > r) r = a[i].first/a[i].second;
    }
    r++;

    vector<pair <float, int>> s(n);
    int N = 18;
    for (int j = 0; j < N; j++){
        float t;
        float S = 0;
        
        t = (l + r) / 2;

        for (int i = 0; i < n; i++){
            s[i] = make_pair(a[i].first - t*a[i].second, i);
        }
        nth_element(s.begin(), s.begin() + n - k, s.end(), fc);
        for (int i = n - k; i < n; i++){
            S += s[i].first;
        }
        if (S < 0){
            r = t;
        }
        else{
            l = t;
        }
    }


    for (int i = n - k; i < n; i++){
        cout << s[i].second + 1 << " ";
    }
    return 0;
}