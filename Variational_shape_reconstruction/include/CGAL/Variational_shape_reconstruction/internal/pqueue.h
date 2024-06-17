#ifndef PQUEUE_H
#define PQUEUE_H

#include <queue>
#include <optional>
#include <algorithm>
#include <vector>

namespace qem {

template<typename T, typename TM>
class Custom_priority_queue : public std::priority_queue<T, std::vector<T>, TM>
{
public:
    void remove(const T& value)
    {
        auto it = std::find(this->c.begin(), this->c.end(), value);
        if (it != this->c.end())
        {
            this->c.erase(it);
            std::make_heap(this->c.begin(), this->c.end(), this->comp);
        }
    }

    void remove(const std::vector<T>& values)
    {
        int count = 0;
        for (const auto& value : values)
        {
            auto it = std::find(this->c.begin(), this->c.end(), value);
            if (it != this->c.end()) {
                this->c.erase(it);
                count++;
            }
        }

        if (count > 0)
            std::make_heap(this->c.begin(), this->c.end(), this->comp);
    }

    std::optional<T> remove_ftc(const T& value)
    {
        auto it = std::find_if(this->c.begin(), this->c.end(), [&value](const T& elem) {
            return elem == value;
        });

        if (it != this->c.end())
        {
            T removed = *it;
            this->c.erase(it);
            std::make_heap(this->c.begin(), this->c.end(), this->comp);
            return std::optional<T>{removed};
        }

        return std::nullopt; // Indicating failure
    }

    std::vector<T> remove_all_ftc(const T& value)
    {
        std::vector<T> removed;
        auto new_end = std::remove_if(this->c.begin(), this->c.end(), [&value, &removed](const T& elem) {
            if (elem.partially_matches(value))
            {
                removed.push_back(elem);
                return true;
            }
            return false;
            });

        if (new_end != this->c.end())
        {
            this->c.erase(new_end, this->c.end());
            std::make_heap(this->c.begin(), this->c.end(), this->comp);
        }

        return removed;
    }

    bool contains(const T& value) const
    {
        return std::find_if(this->c.begin(), this->c.end(), [&value](const T& elem) {
            return elem == value;
            }) != this->c.end();
    }
};

} // namespace qem

#endif // PQUEUE_H
